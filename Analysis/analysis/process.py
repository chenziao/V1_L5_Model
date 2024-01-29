import numpy as np
import pandas as pd
import xarray as xr
import scipy.signal as ss
import pywt
from scipy.ndimage import gaussian_filter

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
    if not isinstance(group_dims, list):
        group_dims = [group_dims]
        group_ids = {(k, ): v for k, v in group_ids.items()}
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
    ).unstack('group').transpose(*group_dims, 'time')
    return grp_rspk


def unit_spike_rate_to_xarray(spikes_df, time, node_ids,
                              frequeny=False, filt_sigma=0.):
    """Count units spike histogram
    spikes_df: dataframe of node ids and spike times
    time: tuple of (start, stop, step) (ms)
    node_ids: list of id of nodes considered
    frequeny: whether return spike frequency in Hz or count
    filt_sigma: sigma (ms) of Gaussian filter for smoothing
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
    spike_rate, _, _ = np.histogram2d(
        spikes_df['node_ids'], spikes_df['timestamps'], bins=(n_bins, t_bins))
    spike_rate = spike_rate[idx_inv, :]
    if frequeny:
        spike_rate = 1000 / dt * spike_rate
    if filt_sigma:
        filt_sigma = (0, filt_sigma / dt)
        spike_rate = gaussian_filter(spike_rate, filt_sigma)
    return spike_rate


def combine_spike_rate(grp_rspk, dim, variables=None, index=slice(None)):
    """Combine spike rate of neuron groups into a new xarray dataset
    grp_rspk: xarray dataset of group spike rate
    dim: group dimension(s) along which to combine
    variables: list of names of variables to combine
        If not specified, apply to all variables except `population_number`
    index: slice or indices of selected groups to combine. Defaults to all
    """
    if not isinstance(dim, list):
        dim = [dim]
        index = [index]
    elif isinstance(index, slice):
        index = [index] * len(dim)
    grp_rspk = grp_rspk.sel(**dict(zip(dim, index)))
    if variables is None:
        variables = [var for var in grp_rspk if var != 'population_number']
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


def wave_hilbert(x, freq_band, fs, filt_order=2, axis=-1):
    sos = ss.butter(N=filt_order, Wn=freq_band, btype='bandpass', fs=fs, output='sos')
    x_a = ss.hilbert(ss.sosfiltfilt(sos, x, axis=axis), axis=axis)
    return x_a


def wave_cwt(x, freq, fs, bandwidth=1.0, axis=-1):
    wavelet = 'cmor' + str(2 * bandwidth ** 2) + '-1.0'
    x_a = pywt.cwt(x, fs / freq, wavelet=wavelet, axis=axis)[0][0]
    return x_a


def get_waves(da, fs, waves, transform, dim='time', component='amp', **kwargs):
    x = [xr.zeros_like(da) for _ in range(len(waves))]
    axis = da.dims.index(dim)
    comp_funcs = {'amp': np.abs, 'pha': np.angle}
    comp_func = comp_funcs.get(component, comp_funcs['amp'])
    for i, freq in enumerate(waves.values()):
        x_a = transform(da.values, freq, fs, axis=axis, **kwargs)
        x[i][:] = comp_func(x_a)
    x = xr.concat(x, dim=pd.Index(waves.keys(), name='wave'))
    return x
