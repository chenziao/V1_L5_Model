import numpy as np
import pandas as pd
import xarray as xr
import scipy.signal as ss
import pywt
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d

from build_input import get_stim_cycle, T_STOP


def get_stim_windows(on_time, off_time, t_start, t_stop=T_STOP, win_extend=0):
    """Time windows of stimulus cycles
    win_extend: extend the window after off time
    Return: 2d-array of time windows, each row is the start/end (sec) of a cycle
    """
    t_cycle, n_cycle = get_stim_cycle(on_time, off_time, t_start, t_stop)
    window =  np.array([0, on_time + win_extend])
    windows = t_start + window + t_cycle * np.arange(n_cycle)[:, None]
    if windows[-1, 1] > t_stop:
        windows = windows[:-1]
    return windows


def get_stim_cycle_dict(fs, on_time, off_time, t_start, t_stop=T_STOP):
    """Parameters of stimulus cycles"""
    t_cycle, n_cycle = get_stim_cycle(on_time, off_time, t_start, t_stop)
    stim_cycle = dict(
        t_cycle = t_cycle, n_cycle = n_cycle,
        t_start = t_start, on_time = on_time,
        i_start = int(t_start * fs), i_cycle = int(t_cycle * fs)
    )
    return stim_cycle


def get_seg_on_stimulus(x, fs, on_time, off_time,
                        t_start, t=T_STOP, tseg=None):
    """Convert input time series during stimulus on time into time segments
    x: input 1d-array or 2d-array where time is the last axis
    fs: sampling frequency (Hz)
    on_time, off_time: on / off time durations
    t_start, t: start and stop time of the stimulus cycles
        If t is an array of time points, the array size is used to infer stop time
    tseg: time segment length. Defaults to on_time if not specified
    Return:
        x_on: same number of dimensions as input, time segments concatenated
        nfft: number of time steps per segment
        stim_cycle: parameters of stimulus cycles
    """
    x = np.asarray(x)
    in_dim = x.ndim
    if in_dim == 1:
        x = x.reshape(1, x.size)
    t = np.asarray(t)
    t_stop = t.size / fs if t.ndim else t
    if tseg is None:
        tseg = on_time # time segment length for PSD (second)
    stim_cycle = get_stim_cycle_dict(fs, on_time, off_time, t_start, t_stop)

    nfft = int(tseg * fs) # steps per segment
    i_on = int(on_time * fs)
    nseg_cycle = int(np.ceil(i_on / nfft))
    x_on = np.zeros((x.shape[0], stim_cycle['n_cycle'] * nseg_cycle * nfft))
    i_start, i_cycle = stim_cycle['i_start'], stim_cycle['i_cycle']

    for i in range(stim_cycle['n_cycle']):
        m = i_start + i * i_cycle
        for j in range(nseg_cycle):
            xx = x[:, m + j * nfft:m + min((j + 1) * nfft, i_on)]
            n = (i * nseg_cycle + j) * nfft
            x_on[:, n:n + xx.shape[1]] = xx
    if in_dim == 1:
        x_on = x_on.ravel()
    return x_on, nfft, stim_cycle


def get_psd_on_stimulus(x, fs, on_time, off_time,
                        t_start, t=T_STOP, tseg=None, axis=-1):
    x_on, nfft, stim_cycle = get_seg_on_stimulus(
        x, fs, on_time, off_time, t_start, t=t, tseg=tseg)
    f, pxx = ss.welch(x_on, fs=fs, window='boxcar',
                      nperseg=nfft, noverlap=0, axis=axis)
    return f, pxx, stim_cycle


def get_coh_on_stimulus(x, y, fs, on_time, off_time,
                        t_start, t=T_STOP, tseg=None):
    xy = np.array([x, y])
    xy_on, nfft, _ = get_seg_on_stimulus(
        xy, fs, on_time, off_time, t_start, t=t, tseg=tseg)
    f, cxy = ss.coherence(xy_on[0], xy_on[1], fs=fs,
        window='boxcar', nperseg=nfft, noverlap=0)
    return f, cxy


def firing_rate(spikes_df, num_cells=None, time_windows=(0.,), frequency=True):
    """
    Count number of spikes for each cell.
    spikes_df: dataframe of node id and spike times (ms)
    num_cells: number of cells (that determines maximum node id)
    time_windows: list of time windows for counting spikes (second)
    frequency: whether return firing frequency in Hz or just number of spikes
    """
    if not spikes_df['timestamps'].is_monotonic:
        spikes_df = spikes_df.sort_values(by='timestamps')
    if num_cells is None:
        num_cells = spikes_df['node_ids'].max() + 1
    time_windows = 1000. * np.asarray(time_windows).ravel()
    if time_windows.size % 2:
        time_windows = np.append(time_windows, spikes_df['timestamps'].max())
    nspk = np.zeros(num_cells, dtype=int)
    n, N = 0, time_windows.size
    count = False
    for t, i in zip(spikes_df['timestamps'], spikes_df['node_ids']):
        while n < N and t > time_windows[n]:
            n += 1
            count = not count
        if count:
            nspk[i] = nspk[i] + 1
    if frequency:
        nspk = nspk / (total_duration(time_windows) / 1000)
    return nspk


def total_duration(time_windows):
    return np.diff(np.reshape(time_windows, (-1, 2)), axis=1).sum()


def pop_spike_rate(spike_times, time=None, time_points=None, frequeny=False):
    """Count spike histogram
    spike_times: spike times (ms)
    time: tuple of (start, stop, step) (ms)
    time_points: evenly spaced time points. If used, argument `time` is ignored.
    frequeny: whether return spike frequency in Hz or count
    """
    if time_points is None:
        time_points = np.arange(*time)
        dt = time[2]
    else:
        time_points = np.asarray(time_points).ravel()
        dt = (time_points[-1] - time_points[0]) / (time_points.size - 1)
    bins = np.append(time_points, time_points[-1] + dt)
    spike_rate, _ = np.histogram(np.asarray(spike_times), bins)
    if frequeny:
        spike_rate = 1000 / dt * spike_rate
    return spike_rate


def group_spike_rate_to_xarray(spikes_df, time, group_ids,
                               group_dims=['assembly', 'population']):
    """Convert spike times into spike rate of neuron groups in xarray dataset
    spikes_df: dataframe of node ids and spike times
    time: left edges of time bins (ms)
    group_ids: dictionary of {group index: group ids}
    group_dims: dimensions in group index. Defaults to ['assembly', 'population']
    """
    time = np.asarray(time)
    fs = 1000 * (time.size - 1) / (time[-1] - time[0])
    if isinstance(group_dims, str):
        reidx = {group_dims: list(group_ids)}
        group_dims = [group_dims]
        group_ids = {(k, ): v for k, v in group_ids.items()}
    else:
        reidx = {}
    group_index = pd.MultiIndex.from_tuples(group_ids, names=group_dims)
    grp_rspk = xr.Dataset(
        dict(
            spike_rate = (
                ['group', 'time'],
                [pop_spike_rate(
                    spikes_df.loc[spikes_df['node_ids'].isin(ids), 'timestamps'],
                    time_points=time,  frequeny=True
                ) / len(ids) for ids in group_ids.values()]
            ),
            population_number = ('group', [len(ids) for ids in group_ids.values()])
        ),
        coords = {'group': group_index, 'time': time + 1000 / fs / 2},
        attrs = {'fs': fs}
    ).unstack('group', fill_value=0).reindex(**reidx).transpose(*group_dims, 'time')
    return grp_rspk


def unit_spike_rate_to_xarray(spikes_df, time, node_ids,
                              frequeny=False, filt_sigma=0., return_count=False):
    """Count units spike histogram
    spikes_df: dataframe of node ids and spike times
    time: Evenly spaced time points (ms), left edges of time bins
    node_ids: list of id of nodes considered
    frequeny: whether return spike frequency in Hz or count
    filt_sigma: sigma (ms) of Gaussian filter for smoothing
    return_count: whether return spike count in addition, with dtype=int
    Return: 2D spike time histogram (node_ids-by-times)
    """
    idx = np.argsort(node_ids)
    node_ids_sort = np.asarray(node_ids)[idx]
    idx_inv = np.zeros_like(idx)
    idx_inv[idx] = range(idx.size)
    spikes_df = spikes_df.loc[spikes_df['node_ids'].isin(node_ids_sort)]
    time = np.asarray(time)
    dt = (time[-1] - time[0]) / (time.size - 1)
    t_bins = np.append(time, time[-1] + 1/dt)
    n_bins = np.append(node_ids_sort, node_ids_sort[-1])
    spike_count, _, _ = np.histogram2d(
        spikes_df['node_ids'], spikes_df['timestamps'], bins=(n_bins, t_bins))
    spike_count = spike_count[idx_inv, :]
    spike_rate = spike_count.copy() if return_count else spike_count
    if frequeny:
        spike_rate = 1000 / dt * spike_rate
    if filt_sigma:
        filt_sigma = (0, filt_sigma / dt)
        spike_rate = gaussian_filter(spike_rate, filt_sigma)
    return (spike_rate, spike_count.astype(int)) if return_count else spike_rate


def combine_spike_rate(grp_rspk, dim, variables=None, index=slice(None)):
    """Combine spike rate of neuron groups into a new xarray dataset
    grp_rspk: xarray dataset of group spike rate
    dim: group dimension(s) along which to combine
    variables: list of names of variables to combine
        If not specified, apply to all variables except `population_number`
    index: slice or indices of selected groups to combine. Defaults to all
    """
    if isinstance(dim, str):
        dim = [dim]
        index = [index]
    elif isinstance(index, slice):
        index = [index] * len(dim)
    grp_rspk = grp_rspk.sel(**dict(zip(dim, index)))
    if variables is None:
        variables = [var for var in grp_rspk if var != 'population_number']
    elif isinstance(variables, str):
        variables = [variables]
    combined_rspk = xr.Dataset()
    for var in variables:
        rspk_weighted = grp_rspk[var].weighted(grp_rspk.population_number)
        combined_rspk.update({var: rspk_weighted.mean(dim=dim)})
    combined_rspk.update(dict(
        population_number=grp_rspk.population_number.sum(dim=dim)))
    combined_rspk.attrs.update(**grp_rspk.attrs)
    return combined_rspk


def windowed_xarray(da, windows, dim='time',
                    new_coord_name='cycle', new_coord=None):
    """Divide xarray into windows of equal size along an axis
    da: input DataArray
    windows: 2d-array of windows
    dim: dimension along which to divide
    new_coord_name: name of new dimemsion along which to concatenate windows
    new_coord: pandas Index object of new coordinates. Defaults to integer index
    """
    win_da = [da.sel({dim: slice(*w)}) for w in windows]
    n_win = min(x.coords[dim].size for x in win_da)
    idx = {dim: slice(n_win)}
    coords = da.coords[dim].isel(idx).coords
    win_da = [x.isel(idx).assign_coords(coords) for x in win_da]
    if new_coord is None:
        new_coord = pd.Index(range(len(win_da)), name=new_coord_name)
    win_da = xr.concat(win_da, dim=new_coord)
    return win_da


def group_windows(win_da, win_grp_idx={}, win_dim='cycle'):
    """Group windows into a dictionary of DataArrays
    win_da: input windowed DataArrays
    win_grp_idx: dictionary of {window group id: window indices}
    win_dim: dimension for different windows
    Return: dictionaries of {window group id: DataArray of grouped windows}
        win_on / win_off for windows selected / not selected by `win_grp_idx` 
    """
    win_on, win_off = {}, {}
    for g, w in win_grp_idx.items():
        win_on[g] = win_da.sel({win_dim: w})
        win_off[g] = win_da.drop_sel({win_dim: w})
    return win_on, win_off


def average_group_windows(win_da, win_dim='cycle', grp_dim='unique_cycle'):
    """Average over windows in each group and stack groups in a DataArray
    win_da: input dictionary of {window group id: DataArray of grouped windows}
    win_dim: dimension for different windows
    grp_dim: dimension along which to stack average of window groups 
    """
    win_avg = {g: xr.concat([x.mean(dim=win_dim), x.std(dim=win_dim)],
                            pd.Index(('mean_', 'std_'), name='stats'))
               for g, x in win_da.items()}
    win_avg = xr.concat(win_avg.values(), dim=pd.Index(win_avg.keys(), name=grp_dim))
    win_avg = win_avg.to_dataset(dim='stats')
    return win_avg


def get_windowed_data(x, windows, win_grp_idx, dim='time',
                      win_dim='cycle', win_coord=None, grp_dim='unique_cycle'):
    """Apply functions of windowing to data
    x: DataArray
    windows: `windows` for `windowed_xarray`
    win_grp_idx: `win_grp_idx` for `group_windows`
    dim: dimension along which to divide
    win_dim: dimension for different windows
    win_coord: pandas Index object of `win_dim` coordinates
    grp_dim: dimension along which to stack average of window groups.
        If None or empty or False, do not calculate average.
    Return: data returned by three functions,
        `windowed_xarray`, `group_windows`, `average_group_windows`
    """
    x_win = windowed_xarray(x, windows, dim=dim,
                            new_coord_name=win_dim, new_coord=win_coord)
    x_win_onff = group_windows(x_win, win_grp_idx, win_dim=win_dim)
    if grp_dim:
        x_win_avg = [average_group_windows(x, win_dim=win_dim, grp_dim=grp_dim)
                     for x in x_win_onff]
    else:
        x_win_avg = None
    return x_win, x_win_onff, x_win_avg


# cone of influence in frequency for cmorxx-1.0 wavelet
f0 = 2 * np.pi
CMOR_COI = 2 ** -0.5
CMOR_FLAMBDA = 4 * np.pi / (f0 + (2 + f0 ** 2) ** 0.5)
COI_FREQ = 1 / (CMOR_COI * CMOR_FLAMBDA)

def cwt_spectrogram(x, fs, nNotes=6, nOctaves=np.inf, freq_range=(0, np.inf),
                    bandwidth=1.0, axis=-1, detrend=False, normalize=False):
    """Calculate spectrogram using continuous wavelet transform"""
    x = np.asarray(x)
    N = x.shape[axis]
    times = np.arange(N) / fs
    # detrend and normalize
    if detrend:
        x = ss.detrend(x, axis=axis, type='linear')
    if normalize:
        x = x / x.std()
    # Define some parameters of our wavelet analysis. 
    # range of scales (in time) that makes sense
    # min = 2 (Nyquist frequency)
    # max = np.floor(N/2)
    nOctaves = min(nOctaves, np.log2(2 * np.floor(N / 2)))
    scales = 2 ** np.arange(1, nOctaves, 1 / nNotes)
    # cwt and the frequencies used. 
    # Use the complex morelet with bw=2*bandwidth^2 and center frequency of 1.0
    # bandwidth is sigma of the gaussian envelope
    wavelet = 'cmor' + str(2 * bandwidth ** 2) + '-1.0'
    frequencies = pywt.scale2frequency(wavelet, scales) * fs
    scales = scales[(frequencies >= freq_range[0]) & (frequencies <= freq_range[1])]
    coef, frequencies = pywt.cwt(x, scales[::-1], wavelet=wavelet, sampling_period=1 / fs, axis=axis)
    power = np.real(coef * np.conj(coef)) # equivalent to power = np.abs(coef)**2
    # cone of influence in terms of wavelength
    coi = N / 2 - np.abs(np.arange(N) - (N - 1) / 2)
    # cone of influence in terms of frequency
    coif = COI_FREQ * fs / coi
    return power, times, frequencies, coif


def instant_amp_by_cwt(x, fs, axis=-1, **cwt_kwargs):
    """Estimate instantaneous amplitude of signal by continuous wavelet transform"""
    sxx, _, frequencies, _ = cwt_spectrogram(x, fs, axis=axis, **cwt_kwargs)
    amp = np.trapz(sxx, frequencies, axis=0) ** 0.5  # integrate over frequencies
    return amp


def wave_hilbert(x, freq_band, fs, filt_order=2, axis=-1):
    sos = ss.butter(N=filt_order, Wn=freq_band, btype='bandpass', fs=fs, output='sos')
    x_a = ss.hilbert(ss.sosfiltfilt(sos, x, axis=axis), axis=axis)
    return x_a


def wave_cwt(x, freq, fs, bandwidth=1.0, axis=-1):
    wavelet = 'cmor' + str(2 * bandwidth ** 2) + '-1.0'
    x_a = pywt.cwt(x, fs / freq, wavelet=wavelet, axis=axis)[0][0]
    return x_a


def get_waves(da, fs, waves, transform, dim='time', component='amp', **kwargs):
    axis = da.dims.index(dim)
    comp_funcs = {'amp': np.abs, 'pha': np.angle, 'none': None}
    comp_func = comp_funcs.get(component, comp_funcs['none'])
    dtype = complex if comp_func is None else None
    x = [xr.zeros_like(da, dtype=dtype) for _ in range(len(waves))]
    for i, freq in enumerate(waves.values()):
        x_a = transform(da.values, freq, fs, axis=axis, **kwargs)
        x[i][:] = x_a if comp_func is None else comp_func(x_a)
        x = xr.concat(x, dim=pd.Index(waves.keys(), name='wave')).rename('wave_' + component)
    if component == 'both':
        funcs = ['amp', 'pha']
        xs = [xr.zeros_like(x, dtype=float) for _ in range(len(funcs))]
        for i, f in enumerate(funcs):
            xs[i][:] = comp_funcs[f](x)
        x = xr.concat(xs, dim=pd.Index(funcs, name='component'))
    return x


def exponential_spike_filter(spikes, tau, cut_val=1e-3, min_rate=None,
                             normalize=False, last_jump=True, only_jump=False):
    """Filter spike train (boolean/int array) with exponential response
    spikes: spike count array (time bins along the last axis)
    tau: time constant of the exponential decay (normalized by time step)
    cut_val: value at which to cutoff the tail of the exponential response
    min_rate: minimum rate of spike (normalized by sampling rate). Default: 1/(9*tau)
        It ensures the filtered values not less than min_val=exp(-1/(min_rate*tau)).
        It also ensures the jump value not less than 1+min_val.
        Specify min_rate=0 to set min_val to 0.
    normalize: whether normalize response to have integral 1 for filtering
    last_jump: whether return a time series with value at each time point equal
        to the unnormalized filtered value at the last spike (jump value)
    only_jump: whether return jump values only at spike times, 0 at non-spike time
    """
    spikes = np.asarray(spikes).astype(float)
    shape = spikes.shape
    if tau <= 0:
        filtered = spikes
        if only_jump:
            jump = spikes.copy()
        elif last_jump:
            jump = np.ones(shape)
    else:
        spikes = spikes.reshape(-1, shape[-1])
        min_val = np.exp(-9) if min_rate is None else \
            (0 if min_rate <= 0 else np.exp(-1 / min_rate / tau))
        t_cut = int(np.ceil(-np.log(cut_val) * tau))
        response = np.exp(-np.arange(t_cut) / tau)[None, :]
        filtered = ss.convolve(spikes, response, mode='full')
        filtered = np.fmax(filtered[:, :shape[-1]], min_val)
        if only_jump:
            idx = spikes > 0
            jump = np.where(idx, filtered, 0)
            if min_val > 0:
                jump[idx] = np.fmax(jump[idx], 1 + min_val)
        elif last_jump:
            min_val = 1 + min_val
            jump = filtered.copy()
            for jp, spks in zip(jump, spikes):
                idx = np.nonzero(spks)[0].tolist() + [None]
                jp[None:idx[0]] = min_val
                for i in range(len(idx) - 1):
                    jp[idx[i]:idx[i + 1]] = max(jp[idx[i]], min_val)
        if normalize:
            filtered /= np.sum(response)
        filtered = filtered.reshape(shape)
    if last_jump or only_jump:
        jump = jump.reshape(shape)
        filtered = (filtered, jump)
    return filtered


def nid_tspk_to_lil(nid, tspk, N):
    """Convert node id and spike times into list of lists of spike times
    nid: sorted node ids of each spike range from 0 to N - 1
    tspk: sorted spike times with the same size as nid
    N: number of nodes
    """
    n = 0
    idx = [0]
    for i, j in enumerate(list(nid) + [N]):
        while j > n:
            n += 1
            idx.append(i)
        if n >= N:
            break
    return [tspk[i:j] for i, j in zip(idx[:-1], idx[1:])]


def get_windowed_spikes(spikes_df, windows, node_ids):
    """Get list of spike times of each unit in node_ids that fall in time windows"""
    spk_df = spikes_df.loc[spikes_df['node_ids'].isin(node_ids)]
    bin_idx = np.digitize(spk_df['timestamps'], windows.ravel())
    spk_df = spk_df.loc[bin_idx % 2 > 0]
    N = len(node_ids)
    node_nid = pd.Series(range(N), index=node_ids)
    spk_df['node_ids'] = node_nid.loc[spk_df['node_ids']].values
    spk_df = spk_df.sort_values(['node_ids', 'timestamps'])
    tspk = nid_tspk_to_lil(spk_df['node_ids'], spk_df['timestamps'].tolist(), N)
    return tspk


def get_spike_amplitude(amp, time, tspk, axis=-1):
    """Get amplitude at spike times"""
    single = len(tspk) and isinstance(tspk[0], float)
    if single:
        tspk = [tspk]
    amp_interp = interp1d(time, amp, axis=axis, assume_sorted=True)
    spk_amp = [amp_interp(t) for t in tspk]
    if single:
        spk_amp = spk_amp[0]
    return spk_amp


def get_spike_phase(phase, time, tspk, axis=-1, min_pha=0.):
    """Get phase at spike times"""
    single = len(tspk) and isinstance(tspk[0], float)
    if single:
        tspk = [tspk]
    phase_interp = interp1d(time, np.unwrap(phase, axis=axis), axis=axis, assume_sorted=True)
    pi2 = 2 * np.pi
    if min_pha:
        spk_pha = [(phase_interp(t) - min_pha) % pi2 + min_pha for t in tspk]
    else:
        spk_pha = [phase_interp(t) % pi2 for t in tspk]
    if single:
        spk_pha = spk_pha[0]
    return spk_pha

