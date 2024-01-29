import numpy as np
import xarray as xr
import scipy.signal as ss
import matplotlib.pyplot as plt
import sys
import os

import pywt
from fooof import FOOOF
from fooof.sim.gen import gen_aperiodic, gen_model
from matplotlib.patches import FancyArrowPatch, ArrowStyle
from build_input import get_populations
from analysis import utils, process

MODEL_PATH = os.path.join('..', 'Model')
CONFIG = 'config.json'
pop_color_base = {'CP': 'blue', 'CS': 'green', 'FSI': 'red', 'LTS': 'purple'}
pop_color = {p: 'tab:' + clr for p, clr in pop_color_base.items()}
pop_names = list(pop_color.keys())


def raster(pop_spike, pop_color, id_column='node_ids', s=0.01, ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    ymin, ymax = [], []
    for p, spike_df in pop_spike.items():
        if len(spike_df) > 0:
            ids = spike_df[id_column].values
            ax.scatter(spike_df['timestamps'], ids, c=pop_color[p], s=s, label=p)
            ymin.append(ids.min())
            ymax.append(ids.max())
    ax.set_xlim(left=0.)
    ax.set_ylim([np.min(ymin) - 1, np.max(ymax) + 1])
    ax.set_title('Spike Raster Plot')
    ax.legend(loc='upper right', framealpha=0.9, markerfirst=False)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Cell ID')
    return ax


def firing_rate_histogram(pop_fr, pop_color, bins=30, min_fr=None,
                          logscale=False, stacked=True, ax=None):
    if logscale and min_fr is not None:
        pop_fr = {p: np.fmax(fr, min_fr) for p, fr in pop_fr.items()}
    fr = np.concatenate(list(pop_fr.values()))
    if logscale:
        fr = fr[fr > 0]
        bins = np.geomspace(fr.min(), fr.max(), bins + 1)
    else:
        bins = np.linspace(fr.min(), fr.max(), bins + 1)
    pop_names = list(pop_fr.keys())
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if stacked:
        ax.hist(pop_fr.values(), bins=bins, label=pop_names,
                color=[pop_color[p] for p in pop_names], stacked=True)
    else:
        for p, fr in pop_fr.items():
            ax.hist(fr, bins=bins, label=p, color=pop_color[p], alpha=0.5)
    if logscale:
        ax.set_xscale('log')
        plt.draw()
        xt = ax.get_xticks()
        xtl = [x.get_text() for x in ax.get_xticklabels()]
        xt = np.append(xt, min_fr)
        xtl.append('0')
        ax.set_xticks(xt)
        ax.set_xticklabels(xtl)
    ax.set_xlim(bins[0], bins[-1])
    ax.legend(loc='upper right')
    ax.set_title('Firing Rate Histogram')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Count')
    return ax


def xcorr_coeff(x, y, max_lag=None, dt=1., plot=True, ax=None):
    x = np.asarray(x)
    y = np.asarray(y)
    x = (x - x.mean()) / x.std()
    y = (y - y.mean()) / y.std()
    xcorr = ss.correlate(x, y) / max(x.size, y.size)
    xcorr_lags = dt * ss.correlation_lags(x.size, y.size)
    if max_lag is not None:
        lag_idx = np.nonzero(np.abs(xcorr_lags) <= max_lag)[0]
        xcorr = xcorr[lag_idx]
        xcorr_lags = xcorr_lags[lag_idx]

    if plot:
        if ax is None:
            _, ax = plt.subplots(1, 1)
        ax.plot(xcorr_lags, xcorr)
        ax.axvline(0., color='gray')
        ax.set_xlabel('Lag')
        ax.set_ylabel('Cross Correlation')
    return xcorr, xcorr_lags


def plot_stimulus_cycles(t, x, stim_cycle, dv_n_sigma=5., var_label='LFP (mV)',
                         fontsize=10, xytext=(.3, 0), ax=None):
    t_cycle, n_cycle = stim_cycle['t_cycle'], stim_cycle['n_cycle']
    t_start, on_time = stim_cycle['t_start'], stim_cycle['on_time']
    i_start, i_cycle = stim_cycle['i_start'], stim_cycle['i_cycle']
    dv = dv_n_sigma * np.std(x[i_start:])
    r_edge = 1000 * t_cycle
    
    if ax is None:
        _, ax = plt.subplots(1, 1)
    ax.plot(t[:i_start], x[:i_start] + n_cycle * dv, 'k')
    ax.annotate('pre stimulus', (max(r_edge, x[i_start]), n_cycle * dv), color='k',
                fontsize=fontsize, xytext=xytext, textcoords='offset fontsize')
    for i in range(n_cycle):
        m = i_start + i * i_cycle
        offset = (n_cycle - i - 1) * dv
        xx = x[m:m + i_cycle] + offset
        h = ax.plot(t[:len(xx)], xx)
        ax.annotate(f'stimulus {i + 1:d}', (r_edge, offset), color=h[0].get_color(),
                    fontsize=fontsize, xytext=xytext, textcoords='offset fontsize')
    ax.axvline(on_time * 1000, color='gray', label='stimulus off')
    ax.set_xlim(0, max(1.15 * r_edge, 1000 * t_start))
    ax.set_ylim(np.array((-1.5, n_cycle + 1.5)) * dv)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel(var_label)


def fit_fooof(f, pxx, aperiodic_mode='fixed', dB_threshold=3., max_n_peaks=10,
              freq_range=None, peak_width_limits=None, report=False,
              plot=False, plt_log=False, plt_range=None, figsize=None):
    if aperiodic_mode != 'knee':
        aperiodic_mode = 'fixed'
    def set_range(x, upper=f[-1]):        
        x = np.array(upper) if x is None else np.array(x)
        return [f[2], x.item()] if x.size == 1 else x.tolist()
    freq_range = set_range(freq_range)
    peak_width_limits = set_range(peak_width_limits, np.inf)

    # Initialize a FOOOF object
    fm = FOOOF(peak_width_limits=peak_width_limits, min_peak_height=dB_threshold / 10,
               peak_threshold=0., max_n_peaks=max_n_peaks, aperiodic_mode=aperiodic_mode)
    # Fit the model
    try:
        fm.fit(f, pxx, freq_range)
    except Exception as e:
        fl = np.linspace(f[0], f[-1], int((f[-1] - f[0]) / np.min(np.diff(f))) + 1)
        fm.fit(fl, np.interp(fl, f, pxx), freq_range)
    results = fm.get_results()

    if report:
        fm.print_results()
        if aperiodic_mode=='knee':
            ap_params = results.aperiodic_params
            if ap_params[1] <= 0:
                print('Negative value of knee parameter occurred. Suggestion: Fit without knee parameter.')
            knee_freq = np.abs(ap_params[1]) ** (1 / ap_params[2])
            print(f'Knee location: {knee_freq:.2f} Hz')
    if plot:
        plt_range = set_range(plt_range)
        fm.plot(plt_log=plt_log)
        plt.xlim(np.log10(plt_range) if plt_log else plt_range)
        if figsize:
            plt.gcf().set_size_inches(figsize)
        plt.show()
    return results, fm


def plot_channel_psd(psd, channel_id=None, plt_range=(0, 100.),
                     aperiodic_mode='knee', freq_range=200., plt_log=True,
                     figsize=(5, 4), **fooof_params):
    """Plot PSD at given chennel with FOOOF results"""
    plt_range = np.array(plt_range)
    if plt_range.size == 1:
        plt_range = (0, plt_range.item())
    psd_plt = psd.sel(frequency=slice(*plt_range))
    plt.rcParams['axes.prop_cycle'] = plt.cycler('color',
        plt.cm.get_cmap('plasma')(np.linspace(0, 1, psd.coords['channel'].size)))
    plt.figure(figsize=figsize)
    plt.plot(psd_plt.frequency, psd_plt.values.T, label=psd_plt.channel.values)
    plt.xlim(plt_range)
    plt.yscale('log')
    plt.legend(loc='upper right', framealpha=0.2, title='channel ID')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    fig1 = plt.gcf()

    if channel_id is None:
        channel_id = psd.channel[0].item()
    print(f'Channel: {channel_id: d}')
    psd_plt = psd.sel(channel=channel_id)
    results = fit_fooof(psd_plt.frequency.values, psd_plt.values,
                        aperiodic_mode=aperiodic_mode, freq_range=freq_range,
                        report=True, plt_log=plt_log, **fooof_params,
                        plot=True, plt_range=plt_range[1], figsize=figsize)
    fig2 = plt.gcf()
    return results, fig1, fig2


def psd_residual(f, pxx, fooof_result, plot=False, plt_log=False, plt_range=None, ax=None):
    full_fit, _, ap_fit = gen_model(f[1:], fooof_result.aperiodic_params,
                                    fooof_result.gaussian_params, return_components=True)
    full_fit, ap_fit = 10 ** full_fit, 10 ** ap_fit

    res_psd = np.insert(pxx[1:] - ap_fit, 0, 0.)
    res_fit = np.insert(full_fit - ap_fit, 0, 0.)

    if plot:
        if ax is None:
            _, ax = plt.subplots(1, 1)
        plt_range = np.array(f[-1]) if plt_range is None else np.array(plt_range)
        if plt_range.size == 1:
            plt_range = [f[1] if plt_log else 0., plt_range.item()]
        f_idx = (f >= plt_range[0]) & (f <= plt_range[1])
        ax.plot(f[f_idx], res_psd[f_idx], 'b', label='residual')
        ax.plot(f[f_idx], res_fit[f_idx], 'r', label='fit')
        if plt_log:
            ax.set_xscale('log')
        ax.set_xlim(plt_range)
        ax.legend(loc='upper right')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('PSD Residual')
    return res_psd, res_fit


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


def cwt_spectrogram_xarray(x, fs, time=None, axis=-1, downsample_fs=None,
                           channel_coords=None, **cwt_kwargs):
    """Calculate spectrogram using continuous wavelet transform and return an xarray.Dataset
    x: input array
    fs: sampling frequency (Hz)
    axis: dimension index of time axis in x
    downsample_fs: downsample to the frequency if specified
    time_unit: unit of time in seconds
    channel_coords: dictionary of {coordinate name: index} for channels
    cwt_kwargs: keyword arguments for cwt_spectrogram()
    """
    x = np.asarray(x)
    T = x.shape[axis] # number of time points
    t = np.arange(T) / fs if time is None else np.asarray(time)
    if downsample_fs is None or downsample_fs >= fs:
        downsample_fs = fs
        downsampled = x
    else:
        num = int(T * downsample_fs / fs)
        downsample_fs = num / T * fs
        downsampled, t = ss.resample(x, num=num, t=t, axis=axis)
    downsampled = np.moveaxis(downsampled, axis, -1)
    sxx, _, f, coif = cwt_spectrogram(downsampled, downsample_fs, **cwt_kwargs)
    sxx = np.moveaxis(sxx, 0, -2) # shape (... , freq, time)
    if channel_coords is None:
        channel_coords = {f'dim_{i:d}': range(d) for i, d in enumerate(sxx.shape[:-2])}
    sxx = xr.DataArray(sxx, coords={**channel_coords, 'frequency': f, 'time': t}).to_dataset(name='PSD')
    sxx.update(dict(cone_of_influence_frequency=xr.DataArray(coif, coords={'time': t})))
    return sxx


def spectrogram_xarray(x, fs, tseg, axis=-1, tres=np.inf, channel_coords=None):
    x = np.asarray(x)
    nfft = int(tseg * fs) # steps per segment
    noverlap = int(max(tseg - tres, 0) * fs)
    f, t, sxx = ss.spectrogram(np.moveaxis(x, axis, -1), fs=fs,
                               nperseg=nfft, noverlap=noverlap, window='boxcar')
    if channel_coords is None:
        channel_coords = {f'dim_{i:d}': range(d) for i, d in enumerate(sxx.shape[:-2])}
    sxx = xr.DataArray(sxx, coords={**channel_coords, 'frequency': f, 'time': t}).to_dataset(name='PSD')
    return sxx


def plot_spectrogram(sxx_xarray, remove_aperiodic=None, log_power=False,
                     plt_range=None, clr_freq_range=None, ax=None):
    """Plot spectrogram. Determine color limits using value in frequency band clr_freq_range"""
    sxx = sxx_xarray.PSD.values.copy()
    t = sxx_xarray.time.values.copy()
    f = sxx_xarray.frequency.values.copy()

    cbar_label = 'PSD' if remove_aperiodic is None else 'PSD Residual'
    if log_power:
        with np.errstate(divide='ignore'):
            sxx = np.log10(sxx)
        cbar_label += ' log(power)'

    if remove_aperiodic is not None:
        f1_idx = 0 if f[0] else 1
        ap_fit = gen_aperiodic(f[f1_idx:], remove_aperiodic.aperiodic_params)
        sxx[f1_idx:, :] -= (ap_fit if log_power else 10 ** ap_fit)[:, None]
        sxx[:f1_idx, :] = 0.

    if ax is None:
        _, ax = plt.subplots(1, 1)
    plt_range = np.array(f[-1]) if plt_range is None else np.array(plt_range)
    if plt_range.size == 1:
        plt_range = [f[0 if f[0] else 1] if log_power else 0., plt_range.item()]
    f_idx = (f >= plt_range[0]) & (f <= plt_range[1])
    if clr_freq_range is None:
        vmin, vmax = None, None
    else:
        c_idx = (f >= clr_freq_range[0]) & (f <= clr_freq_range[1])
        vmin, vmax = sxx[c_idx, :].min(), sxx[c_idx, :].max()

    f = f[f_idx]
    pcm = ax.pcolormesh(t, f, sxx[f_idx, :], shading='gouraud', vmin=vmin, vmax=vmax)
    if 'cone_of_influence_frequency' in sxx_xarray:
        coif = sxx_xarray.cone_of_influence_frequency
        ax.plot(t, coif)
        ax.fill_between(t, coif, step='mid', alpha=0.2)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(f[0], f[-1])
    plt.colorbar(mappable=pcm, ax=ax, label=cbar_label)
    ax.set_xlabel('Time (sec)')
    ax.set_ylabel('Frequency (Hz)')
    return sxx


ARROWSTYLE = ArrowStyle('->, head_length=1, head_width=0.5')
ARROWPROPS = dict(shrinkA=0, shrinkB=0, mutation_scale=10)

def trajectory_pairplot(data, time=None, xlabels=None, ylabels=None,
                        marker_times=[], marker_names=[], marker_props={},
                        traj_props={}, diag_props={}, arrow_props={},
                        arrow_loc=(0.45, 0.55), figsize=(3, 2.5)):
    time = next(data.values()).time.values if time is None else np.asarray(time)
    xlabels = list(data.keys()) if xlabels is None else list(xlabels)
    ylabels = xlabels if ylabels is None else list(ylabels)
    if not all([L in data for L in xlabels + ylabels]):
        print(set(xlabels + ylabels) - data.keys())
        raise ValueError("The above labels not found in data")
    data = {k: np.asarray(v) for k, v in data.items()}
    traj_props = {'color': 'b', **traj_props}
    diag_props = {'color': 'r', **diag_props}
    arrow_props = {'color': traj_props['color'], **arrow_props}

    marker_times = np.clip(marker_times, time[0], time[-1])
    if len(marker_names) != marker_times.size:
        raise ValueError("The size of marker_names does not match marker_names")
    marker_props = {'linestyle': 'none', 'marker': 'o', 'markersize': 10,
                    'markerfacecolor': 'none', **marker_props}
    for k, v in marker_props.items():
        if isinstance(v, (tuple, list)):
            if len(v) != marker_times.size:
                raise ValueError(f"The length of property {k:s} does not match"
                                 " the number of markers")
        else:
            marker_props[k] = [v] * marker_times.size
    marker_props = [{k: v[m] for k, v in marker_props.items()}
                    for m in range(marker_times.size)]
    marker_locs = {k: np.interp(marker_times, time, v) for k, v in data.items()}
    frac = np.asarray(arrow_loc)
    arrow_times = np.sort(marker_times)[:, None]
    arrow_times = frac * arrow_times[1:] + (1 - frac) * arrow_times[:-1]
    arrow_locs = {k: np.interp(arrow_times, time, v) for k, v in data.items()}

    nrow, ncol = len(ylabels), len(xlabels)
    _, axs = plt.subplots(nrow, ncol, squeeze=False,
                        figsize=(ncol * figsize[0], nrow * figsize[1]))
    xlim, ylim = np.empty((ncol, 2)), np.empty((nrow, 2))
    xlim[:, 0], xlim[:, 1] = -np.inf, np.inf
    ylim[:, 0], ylim[:, 1] = -np.inf, np.inf
    for i, yl in enumerate(ylabels):
        y = data[yl]
        my = marker_locs[yl]
        a_y = arrow_locs[yl]
        for j, xl in enumerate(xlabels):
            x = data[xl]
            ax = axs[i, j]
            if xl == yl:
                mx = marker_times
                ax.plot(time, y, **diag_props)
                ax.text(.5, 0.02,'Time (ms)', transform=ax.transAxes,
                        horizontalalignment='center', verticalalignment='bottom')
            else:
                mx = marker_locs[xl]
                a_x = arrow_locs[xl]
                ax.plot(x, y, **traj_props)
                for k in range(len(arrow_times)):
                    ax.add_patch(FancyArrowPatch(*np.vstack((a_x[k], a_y[k])).T,
                        arrowstyle=ARROWSTYLE, **ARROWPROPS, **arrow_props))
                xlim[j] = np.clip(xlim[j], *ax.get_xlim())
                ylim[i] = np.clip(ylim[i], *ax.get_ylim())
            for m in range(marker_times.size):
                ax.plot(mx[m], my[m], label=marker_names[m], **marker_props[m])

    for xl, ax in zip(xlabels, axs[-1, :]):
        ax.set_xlabel(xl)
    for yl, ax in zip(ylabels, axs[:, 0]):
        ax.set_ylabel(yl)
    for i, yl in enumerate(ylabels):
        for j, xl in enumerate(xlabels):
            axs[i, j].set_ylim(ylim[i])
            if xl != yl:
                axs[i, j].set_xlim(xlim[j])
    axs[0, 0].legend(loc='upper right', framealpha=0.2)
    plt.tight_layout()
    return axs


def plot(choose, spike_file=None, config=None, figsize=(6.4, 4.8)):

    global CONFIG
    if config is None:
        config = CONFIG
    else:
        CONFIG = config
    config = os.path.join(MODEL_PATH, config)
    if spike_file is None:
        spike_file = 'output/spikes.h5'

    if choose<=2:
        from bmtool.util.util import load_nodes_from_config

        network_name = 'cortex'
        spikes_df = utils.load_spikes_to_df(spike_file, network_name)
        node_df = load_nodes_from_config(config)[network_name]

    if choose==1:
        spikes_df['pop_name'] = node_df.loc[spikes_df['node_ids'], 'pop_name'].values
        pop_spike = get_populations(spikes_df, pop_names)

        print("Plotting cortex spike raster")
        _, ax = plt.subplots(1, 1, figsize=figsize)
        raster(pop_spike, pop_color, ax=ax)
        plt.show()
        return pop_spike

    elif choose==2:
        from bmtk.simulator import bionet

        conf = bionet.Config.from_json(config)
        t_stop = conf['run']['tstop'] / 1000

        frs = process.firing_rate(spikes_df, num_cells=len(node_df), time_windows=(0., t_stop))
        pop_ids = get_populations(node_df, pop_names, only_id=True)
        pop_fr = {p: frs[nid] for p, nid in pop_ids.items()}

        print('Firing rate: mean, std')
        for p, fr in pop_fr.items():
            print(f'{p}: {fr.mean():.3g}, {fr.std():.3g}')

        print("Plotting firing rates")
        min_fr = 0.5 / process.total_duration((0., t_stop)) # to replace 0 spikes
        _, ax = plt.subplots(1, 1, figsize=figsize)
        firing_rate_histogram(pop_fr, pop_color, bins=20, min_fr=min_fr,
                              logscale=True, stacked=False, ax=ax)
        plt.show()
        return frs

    elif choose==3:
        from bmtk.analyzer.compartment import plot_traces
        print("plotting voltage trace")
        _ = plot_traces(config_file=config, report_name='v_report',
                        group_by='pop_name', average=True)
    else:
        print('Choice ID not identified.')


if __name__ == '__main__':
    argv = sys.argv[1:]
    narg = len(argv)
    if narg > 0:
        argv[0] = int(argv[0])
    plot(*argv)