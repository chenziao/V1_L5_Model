import os
import csv
import json
import time
import argparse

import numpy as np
import pandas as pd
from bmtool.util import util
from bmtk.utils.reports.spike_trains import PoissonSpikeGenerator


INPUT_PATH = "./input"
STIMULUS = ['baseline', 'short', 'long']

N_ASSEMBLIES = 9  # number of assemblies
NET_SEED = 4321  # random seed for network r.v.'s (e.g. assemblies, firing rate)
PSG_SEED = 0  # poisson spike generator random seed for different trials
rng = np.random.default_rng(NET_SEED)

T_STOP = 28.  # sec. Simulation time
T_START = 1.0  # sec. Time to start burst input
t_start = T_START
on_time = 1.0  # sec. Burst input duration
off_time = 0.5  # sec. Silence duration
off_time_expr = 1.0  # sec. Silence duration for experiments (longer for reset)
n_cycles_expr = 10  # number of cycles for experiments

SHELL_FR = {
    'CP': (1.9, 1.3),
    'CS': (1.35, 1.2),
    'FSI': (4.1, 6.0),
    'LTS': (0.2, 0.8)
}  # firing rate of shell neurons (mean, stdev)
SHELL_FR = pd.DataFrame.from_dict(
    SHELL_FR, orient='index', columns=('mean', 'stdev')).rename_axis('pop_name')
SHELL_CONSTANT_FR = False  # whether use constant firing rate for shell neurons


def num_prop(ratio, N):
    """Calculate numbers of total N in proportion to ratio"""
    ratio = np.asarray(ratio)
    p = np.cumsum(np.insert(ratio.ravel(), 0, 0))  # cumulative proportion
    return np.diff(np.round(N / p[-1] * p).astype(int)).reshape(ratio.shape)


def lognormal(mean, stdev, size=None):
    """Generate random values from lognormal given mean and stdev"""
    sigma2 = np.log((stdev / mean) ** 2 + 1)
    mu = np.log(mean) - sigma2 / 2
    sigma = sigma2 ** 0.5
    return rng.lognormal(mu, sigma, size)


def psg_lognormal_fr(psg, node_ids, mean, stdev, times):
    """Generate lognormal distributed firing rate for each node independently.
    Then add the firing rate of each node to the given PoissonSpikeGenerator.
    """
    firing_rates = lognormal(mean, stdev, len(node_ids))
    for node_id, fr in zip(node_ids, firing_rates):
        psg.add(node_ids=node_id, firing_rate=fr, times=times)
    return firing_rates


def df2node_id(df):
    """Get node ids from a node dataframe into a list"""
    return df.index.tolist()


def get_pop(node_df, pop_name):
    """Get nodes with given population name from the nodes dataframe"""
    return node_df.loc[node_df['pop_name'] == pop_name]


def get_pop_id(node_df, pop_name):
    """Get node ids with given population name from the nodes dataframe"""
    return df2node_id(get_pop(node_df, pop_name))


def get_populations(node_df, pop_names, only_id=False):
    """Get node dataframes of multiple populations from the nodes dataframe"""
    func = get_pop_id if only_id else get_pop
    return {p: func(node_df, p) for p in pop_names}


def get_assembly(Thal_nodes, Cortex_nodes, n_assemblies):
    """Divide PNs into n_assemblies and return lists of ids in each assembly"""
    CP_nodes, CS_nodes = Cortex_nodes['CP'], Cortex_nodes['CS']
    num_CP, num_CS = len(CP_nodes), len(CS_nodes)
    num_PN = len(Thal_nodes)
    if num_CP + num_CS != num_PN:
        raise ValueError("Number of thalamus cells don't match number of PNs")

    n_per_assemb = num_prop(np.ones(n_assemblies), num_PN)
    split_idx = np.cumsum(n_per_assemb)[:-1]  # indices at which to split
    assemb_idx = rng.permutation(num_PN)  # random shuffle for assemblies
    assemb_idx = np.split(assemb_idx, split_idx)  # split into assemblies

    Thal_ids = np.array(Thal_nodes)
    PN_ids = np.array(CP_nodes + CS_nodes)
    Thal_assy = []
    PN_assy = []
    for idx in assemb_idx:
        idx = np.sort(idx)
        Thal_assy.append(Thal_ids[idx])
        PN_assy.append(PN_ids[idx])
    return Thal_assy, PN_assy


def input_pairs_to_file(file, source, target):
    """Save ids of input source/target pairs to file. Rows are source ids for
    each population followed by target ids for each population.
    """
    with open(file, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(source)
        writer.writerows(target)


def input_pairs_from_file(file, pop_index=None):
    """Load ids of input source/target pairs from file"""
    with open(file, 'r') as f:
        ids = [np.array(row, dtype='uint64') for row in csv.reader(f)]
    n_assemblies = len(ids) // 2
    source = ids[:n_assemblies]
    target = ids[n_assemblies:]
    if pop_index is not None:
        if hasattr(pop_index, '__len__'):
            source = [source[i] for i in pop_index]
            target = [target[i] for i in pop_index]
        else:
            source = source[pop_index]
            target = target[pop_index]
    return source, target


def get_stim_cycle(on_time=on_time, off_time=off_time,
                   t_start=T_START, t_stop=T_STOP):
    """Get burst input stimulus parameters, (duration, number) of cycles.
    Poisson input is first on for on_time starting at t_start, then off for
    off_time. This repeats until the last on_time can complete before t_stop.
    """
    t_cycle = on_time + off_time
    n_cycle = int(np.floor((t_stop + off_time - t_start) / t_cycle))
    return t_cycle, n_cycle


def get_psg_from_fr(psg, source_assy, params):
    """Add firing rate traces to PoissonSpikeGenerator object
    psg: PoissonSpikeGenerator object
    source_assy: list of node ids in each source assembly
    params: list of argument dictionaries with keys `firing_rate` and `times`
    """
    for ids, kwargs in zip(source_assy, params):
        psg.add(node_ids=ids, **kwargs)
    return psg


def plot_fr_traces(params, figsize=(10, 2), **line_kwargs):
    """Plot firing rate traces from parameters for PoissonSpikeGenerator"""
    import matplotlib.pyplot as plt
    n_assemblies = len(params)
    fig, axs = plt.subplots(n_assemblies, 1, squeeze=False,
                            figsize=(figsize[0], n_assemblies * figsize[1]))
    kwargs = dict(marker='o', markerfacecolor='none')
    kwargs.update(line_kwargs)
    for i, ax in enumerate(axs.ravel()):
        p = params[i]
        t = p['times']
        ax.plot(t, p['firing_rate'], **kwargs)
        ax.set_ylabel('Firing rate (Hz)')
        ax.set_xlim(t[0], t[-1])
        ax.set_ylim(bottom=0.)
        ax.set_title(f'Assembly {i:d}')
    ax.set_xlabel('Time (sec)')
    plt.tight_layout()
    return fig, axs


def get_fr_short(n_assemblies, firing_rate=(0., 0.),
                 on_time=on_time, off_time=off_time,
                 t_start=T_START, t_stop=T_STOP):
    """Short burst is delivered to each assembly sequentially within each cycle.
    n_assemblies: number of assemblies
    firing_rate: 2-tuple of firing rate at off and on time, respectively
        Pad zero to the beginning if sequence is shorter than required
    t_start, t_stop: start and stop time of the stimulus cycles
    Return: firing rate traces
    """
    firing_rate = np.asarray(firing_rate).ravel()[:2]
    firing_rate = np.concatenate((np.zeros(2 - firing_rate.size), firing_rate))
    t_cycle, n_cycle = get_stim_cycle(on_time, off_time, t_start, t_stop)

    times = np.empty((n_assemblies, n_cycle * 4 + 2))
    times[:, 0] = 0.
    times[:, -1] = t_stop
    on_times = np.linspace(0, on_time, n_assemblies + 1)
    for j in range(n_cycle):
        ts = t_start + t_cycle * j + on_times
        for i in range(n_assemblies):
            times[i, j * 4 + 1:j * 4 + 5] = np.repeat(ts[i:i + 2], 2)
    fr = np.append(np.tile(firing_rate, n_cycle), firing_rate[0])
    fr = np.repeat(fr, 2)

    params = [dict(firing_rate=fr, times=times[i]) for i in range(n_assemblies)]
    return params


def get_fr_long(n_assemblies, firing_rate=(0., 0.),
                on_time=on_time, off_time=off_time,
                t_start=T_START, t_stop=T_STOP):
    """Long burst is delivered to one assembly in each cycle.
    n_assemblies: number of assemblies
    firing_rate: 2-tuple of firing rate at off and on time, respectively
        Pad zero to the beginning if sequence is shorter than required
    on_time, off_time: on / off time durations
    t_start, t_stop: start and stop time of the stimulus cycles
    Return: firing rate traces
    """
    firing_rate = np.asarray(firing_rate).ravel()[:2]
    firing_rate = np.concatenate((np.zeros(2 - firing_rate.size), firing_rate))

    firing_rate = np.append(firing_rate, firing_rate[1])
    params = get_fr_loop(n_assemblies, firing_rate,
                         on_times=on_time, off_time=off_time,
                         t_start=t_start, t_stop=t_stop)
    return params


def get_fr_ramp(n_assemblies, firing_rate=(0., 0., 0.),
                on_time=on_time, off_time=off_time,
                ramp_on_time=None, ramp_off_time=None,
                t_start=T_START, t_stop=T_STOP):
    """Ramping input is delivered to one assembly in each cycle.
    n_assemblies: number of assemblies
    firing_rate: 3-tuple of firing rate at off time, start and end of on time
        Pad zero to the beginning if sequence is shorter than required
    on_time, off_time: on / off time durations
    ramp_on_time, ramp_off_time: start and end time of ramp in on time duration
        Firing rate is constant before start and after end of ramp time.
    t_start, t_stop: start and stop time of the stimulus cycles
    Return: firing rate traces
    """
    firing_rate = np.asarray(firing_rate).ravel()[:3]
    firing_rate = np.concatenate((np.zeros(3 - firing_rate.size), firing_rate))

    firing_rate = np.repeat(firing_rate, 2)[1:]
    ramp_off_time = on_time if ramp_off_time is None else min(ramp_off_time, on_time)
    ramp_on_time = 0. if ramp_on_time is None else min(ramp_on_time, ramp_off_time)
    on_times = (0., ramp_on_time, ramp_off_time, on_time)
    params = get_fr_loop(n_assemblies, firing_rate=firing_rate,
                         on_times=on_times, off_time=off_time,
                         t_start=t_start, t_stop=t_stop)
    return params


def get_fr_join(n_assemblies, firing_rate=(0., 0.),
                on_time=on_time, off_time=off_time,
                quit=False, ramp_on_time=None, ramp_off_time=None,
                t_start=T_START, t_stop=T_STOP, n_steps=20):
    """Input is delivered to an increasing portion of one assembly in each cycle.
    n_assemblies: number of assemblies
    firing_rate: 2-tuple of firing rate at off and on time, respectively
        Pad zero to the beginning if sequence is shorter than required
    on_time, off_time: on / off time durations
    t_start, t_stop: start and stop time of the stimulus cycles
    n_steps: number of steps to divide up each assembly. By each step, an equal
        portion of neurons in each assembly join to receive input.
    Return: firing rate traces (of all n_steps in each assembly)
    """
    firing_rate = np.asarray(firing_rate).ravel()[:2]
    firing_rate = np.concatenate((np.zeros(2 - firing_rate.size), firing_rate))

    fr = firing_rate[1]
    firing_rate = np.full(5, firing_rate[0])
    firing_rate[slice(1, 3) if quit else slice(3, 5)] = fr
    ramp_off_time = on_time if ramp_off_time is None else min(ramp_off_time, on_time)
    ramp_on_time = 0. if ramp_on_time is None else min(ramp_on_time, ramp_off_time)
    t_offset = np.linspace(ramp_on_time, ramp_off_time, n_steps, endpoint=False)
    params = [fr for t in (t_offset[::-1] if quit else t_offset)
              for fr in get_fr_loop(n_assemblies, firing_rate=firing_rate, 
                  on_times=(0., t, t, on_time), off_time=off_time,
                  t_start=t_start, t_stop=t_stop)]
    params = [fr for i in range(n_assemblies) for fr in params[i::n_assemblies]]
    return params


def get_join_split(size_assemblies, n_steps=20,
                   low_portion=0., high_portion=1., seed=None):
    """Split each assembly into equal steps for join stimulus
    size_assemblies: list of size of each of all assemblies
    n_steps: number of steps to divide up each assembly
    low_portion, high_portion: the lowest and highest proportion of neurons
        in an assembly stimulated during a stimulus cycle
    seed: random seed for the split. If not specified, join in original order
    Return: nested lists of indices in each step in each assembly
    """
    ratio = np.full(n_steps, (high_portion - low_portion) / n_steps)
    ratio[0] += low_portion
    split_ids = []
    if seed is not None:
        rng_tmp = np.random.default_rng(seed)  # shuffle ids in each assembly
    for n in size_assemblies:
        assy_ids = np.arange(n) if seed is None else rng_tmp.permutation(n)  
        n_per_step = num_prop(ratio, n * high_portion)  # split into steps
        split_idx = np.cumsum(n_per_step)  # indices at which to split
        split_ids.append(np.split(assy_ids, split_idx)[:-1])
    if seed is not None:
        split_ids = [[np.sort(i) for i in idx] for idx in split_ids]
    return split_ids


def split_join_assemblies(node_ids, split_ids):
    """Split the node ids for join stimulus and concatenate in a single list"""
    join_ids = []
    for ids, idx in zip(node_ids, split_ids):
        for i in idx:
            join_ids.append(np.asarray(ids)[np.asarray(i)])
    return join_ids


def get_fr_fade(n_assemblies, firing_rate=(0., 0., 0., 0., 0.),
                on_time=on_time, off_time=off_time,
                ramp_on_time=None, ramp_off_time=None,
                t_start=T_START, t_stop=T_STOP):
    """Input fades in and out between a pair of assemblies in each cycle.
    n_assemblies: number of assembly pairs
    firing_rate: 5-tuple of firing rate at off time, start and end of on time
        of the fade out assembly, followed by that of the fade in assembly.
        Mirror the last two entries to the front of them or pad zeros if
        sequence is shorter than required.
    on_time, off_time: on / off time durations
    ramp_on_time, ramp_off_time: start and end time of ramp in on time duration
        Firing rate is constant before start and after end of ramp time.
    t_start, t_stop: start and stop time of the stimulus cycles
    Return: firing rate traces
    """
    firing_rate = np.asarray(firing_rate).ravel()
    n_fr = firing_rate.size
    firing_rate = firing_rate[:5]
    firing_rate = np.concatenate((np.zeros(5 - firing_rate.size), firing_rate))
    if n_fr <= 3:
        firing_rate[1] = firing_rate[4]
    if n_fr == 2:
        firing_rate[2] = firing_rate[3]

    ramp_params = dict(
        n_assemblies=n_assemblies, on_time=on_time, off_time=off_time,
        ramp_on_time=ramp_on_time, ramp_off_time=ramp_off_time,
        t_start=t_start, t_stop=t_stop
    )
    params_out = get_fr_ramp(firing_rate=firing_rate[[0, 1, 2]], **ramp_params)
    params_in = get_fr_ramp(firing_rate=firing_rate[[0, 3, 4]], **ramp_params)
    params = sum(([params_out[i], params_in[i]] for i in range(n_assemblies)), [])
    return params


def get_fr_loop(n_assemblies, firing_rate=(0., 0., 0.),
                on_times=(on_time, ), off_time=off_time,
                t_start=T_START, t_stop=T_STOP):
    """Poisson input is first on for on_time starting at t_start, then off for
    off_time. This repeats until the last on-time can complete before t_stop.
    Same pattern is delivered to one assembly in each cycle.
    n_assemblies: number of assemblies
    firing_rate: tuple of firing rate at off time followed by those at on time
    on_times: time points corresponding to firing rates during on time
        The smallest in on_times should be 0.
        The largest in on_times determines the on time duration.
    off_time: off time duration
    t_start, t_stop: start and stop time of the stimulus cycles
    Return: firing rate traces
    """
    firing_rate = np.asarray(firing_rate).ravel()
    on_times = np.fmax(np.sort(np.asarray(on_times).ravel()), 0)
    if on_times[0]:
        on_times = np.insert(on_times, 0, 0.)
    if firing_rate.size - on_times.size != 1:
        raise ValueError("Length of `firing_rate` should be len(on_times) + 1.")
    t_cycle, n_cycle = get_stim_cycle(on_times[-1], off_time, t_start, t_stop)

    times = [[0] for _ in range(n_assemblies)]
    for j in range(n_cycle):
        ts = t_start + t_cycle * j + on_times
        times[j % n_assemblies].extend(np.insert(ts, [0, -1], ts[[0, -1]]))

    params = []
    fr = []
    fr0 = firing_rate[0]
    for ts in times:
        ts.append(t_stop)
        n = (len(ts) - 2) // (on_times.size + 2)
        if len(fr) != len(ts):
            fr = np.append(np.tile(np.insert(firing_rate, 0, fr0), n), [fr0, fr0])
        params.append(dict(firing_rate=fr, times=ts))
    return params


def get_std_param(stim_setting={}, stimulus='baseline'):
    """Generater parameters for standard stimulus from settings
    stim_setting: dictionary of standard stimulus settings
    stimulus: stimulus type name
    Return: firing rate traces
    """
    p = stim_setting.get(stimulus)
    n_assemblies = stim_setting.get('n_assemblies', N_ASSEMBLIES)
    if stimulus == 'baseline':
        times = (0, p.get('t_stop', T_STOP))
        fr_params = [{'firing_rate': p['PN_firing_rate'], 'times': times},
                     {'firing_rate': p['ITN_firing_rate'], 'times': times}]
    elif stimulus == 'const':
        t_stop = p.get('t_stop', T_STOP)
        fr_params = get_fr_long(n_assemblies, [p['firing_rate']] * 2,
                                on_time=t_stop, off_time=0., t_stop=t_stop)
    elif stimulus == 'short':
        fr_params = get_fr_short(n_assemblies, **p)
    elif stimulus == 'long':
        fr_params = get_fr_long(n_assemblies, **p)
    else:
        raise ValueError("%s is not standard stimulus type" % stimulus)
    return fr_params


def get_ramp_param(stim_setting={}, **add_default_setting):
    """Generater parameters for ramp stimulus from settings 
    stim_setting: dictionary of stimulus settings and parameters
    add_default_setting: additional default parameters
    Return: firing rate traces, stimulus setting and parameters
    """
    default_setting = dict(
        assembly_index = [0],
        n_cycles = n_cycles_expr,
        on_time = on_time,
        off_time = off_time_expr,
        t_start = t_start,
    )
    default_setting.update(add_default_setting)
    setting = {**default_setting, **stim_setting.get('setting', {})}
    assembly_index = setting['assembly_index']

    stim_params_keys = ['firing_rate', 'on_time', 'off_time', 't_start',
                        'ramp_on_time', 'ramp_off_time']
    stim_params = {k: setting[k] for k in stim_params_keys if k in setting}
    stim_params['t_stop'] = setting['t_start'] + setting['n_cycles'] \
        * (setting['on_time'] + setting['off_time'])
    fr_params = get_fr_ramp(len(assembly_index), **stim_params)

    stim_setting = {'setting': setting, 'stim_params': stim_params}
    return fr_params, stim_setting


def get_join_param(stim_setting={}, size_assemblies=[], **add_default_setting):
    """Generater parameters for join stimulus from settings 
    stim_setting: dictionary of stimulus settings and parameters
    size_assemblies: list of size of each of all assemblies
    add_default_setting: additional default parameters
    Return: firing rate traces, stimulus setting and parameters
    """
    default_setting = dict(
        assembly_index = [0],
        n_cycles = n_cycles_expr,
        on_time = on_time,
        off_time = off_time_expr,
        t_start = t_start,
        n_steps = 20,
        seed = None,
    )
    default_setting.update(add_default_setting)
    setting = {**default_setting, **stim_setting.get('setting', {})}
    assembly_index = setting['assembly_index']

    stim_params_keys = ['firing_rate', 'on_time', 'off_time', 'quit',
                        'ramp_on_time', 'ramp_off_time', 't_start', 'n_steps']
    stim_params = {k: setting[k] for k in stim_params_keys if k in setting}
    stim_params['t_stop'] = setting['t_start'] + setting['n_cycles'] \
        * (setting['on_time'] + setting['off_time'])
    fr_params = get_fr_join(len(assembly_index), **stim_params)

    split_params_keys = ['n_steps', 'low_portion', 'high_portion', 'seed']
    split_params = {k: setting[k] for k in split_params_keys if k in setting}
    if max(assembly_index) < len(size_assemblies):
        size_assemblies = np.asarray(size_assemblies)[assembly_index]
        split_ids = get_join_split(size_assemblies, **split_params)
    else:
        split_ids = None

    stim_setting = {'setting': setting, 'stim_params': stim_params,
                    'split_params': split_params}
    return fr_params, stim_setting, split_ids


def get_fade_param(stim_setting={}, **add_default_setting):
    """Generater parameters for fade stimulus from settings 
    stim_setting: dictionary of stimulus settings and parameters
    add_default_setting: additional default parameters
    Return: firing rate traces, stimulus setting and parameters
    """
    default_setting = dict(
        assembly_index = [0, 1],
        n_cycles = n_cycles_expr,
        on_time = 2 * on_time,
        off_time = off_time_expr,
        ramp_on_time = 0.5 * on_time,
        ramp_off_time = 1.5 * on_time,
        t_start = t_start,
    )
    default_setting.update(add_default_setting)
    setting = {**default_setting, **stim_setting.get('setting', {})}
    assembly_index = setting['assembly_index']
    if len(assembly_index) % 2:
        assembly_index = assembly_index[:-1]
    if not assembly_index:
        raise ValueError("Specify at least two assemblies in `assembly_index`")

    stim_params_keys = ['firing_rate', 'on_time', 'off_time', 't_start',
                        'ramp_on_time', 'ramp_off_time']
    stim_params = {k: setting[k] for k in stim_params_keys if k in setting}
    stim_params['t_stop'] = setting['t_start'] + setting['n_cycles'] \
        * (setting['on_time'] + setting['off_time'])
    fr_params = get_fr_fade(len(assembly_index) // 2, **stim_params)

    stim_setting = {'setting': setting, 'stim_params': stim_params}
    return fr_params, stim_setting


def load_stim_file(input_path=INPUT_PATH, stim_file=None, file_name=''):
    """Load stimulus file
    input_path: directory to store updated stimulus file
    stim_file: stimulus json file path to load parameters from
    file_name: default name for stimulus file if `stim_file` not specified
    Return: stimulus settings, updated stimulus file path
    """
    if stim_file is None:
        load_stim_file = None
        stim_file = new_file_name(input_path, file_name, '.json')
    else:
        _, ext = os.path.splitext(stim_file)
        if ext != '.json':
            stim_file += '.json'
        load_stim_file = stim_file if os.path.isfile(stim_file) else None
        stim_file = os.path.join(input_path, os.path.split(stim_file)[1])
        if load_stim_file is None and os.path.isfile(stim_file):
            load_stim_file = stim_file

    if load_stim_file is None:
        stim_setting = {}
        print("Warning: Stimulus file for %s not found. "
              "Using default settings." % file_name)
    else:
        with open(load_stim_file, 'r') as f:
            stim_setting = json.load(f)
    return stim_setting, stim_file


def new_file_name(directory, file_name, ext=''):
    """Get file name with trailing number different from existing files that
    have the same leading name and extension in a directory"""
    file_list = [os.path.splitext(s) for s in os.listdir(directory)]
    file_list = [s[0] for s in file_list if s[1] == ext and file_name in s[0]]
    ids = [s.replace(file_name, '').rsplit('_', 1) for s in file_list]
    ids = [int(s[1]) for s in ids if len(s) == 2 and not s[0] and s[1].isdigit()]
    new_id = max(ids + [-1]) + 1
    return os.path.join(directory, file_name + '_%d' % new_id + ext)


def write_std_stim_file(stim_params={}, input_path=INPUT_PATH,
                        file_name='standard_stimulus.json'):
    if len(stim_params) > 1:
        stim_file = os.path.join(input_path, file_name)
        if os.path.isfile(stim_file):
            with open(stim_file, 'r') as f:
                stim_params = {**json.load(f), **stim_params}
        with open(stim_file, 'w') as f:
            json.dump(stim_params, f, indent=2)


def write_seeds_file(psg_seed=PSG_SEED, net_seed=NET_SEED, stimulus=STIMULUS,
                     input_path=INPUT_PATH, seeds_file_name='random_seeds'):
    seeds_file = os.path.join(input_path, seeds_file_name + '.json')
    if os.path.isfile(seeds_file):
        with open(seeds_file, 'r') as f:
            seeds = json.load(f)
    else:
        seeds = []
    seed = [s for s in seeds if s['net_seed'] == net_seed
                            and s['psg_seed'] == psg_seed]
    if seed:
        seed = seed[0]
        stimulus_new = [s for s in stimulus if s not in seed['stimulus']]
        seed['stimulus'] += stimulus_new
        overwrite = bool(stimulus_new)
    else:
        seed = dict(net_seed=net_seed, psg_seed=psg_seed, stimulus=stimulus)
        seeds.append(seed)
        overwrite = True
    if overwrite:
        with open(seeds_file, 'w') as f:
            json.dump(seeds, f, indent=2)


def build_input(t_stop=T_STOP, t_start=T_START, n_assemblies=N_ASSEMBLIES,
                burst_fr=None, psg_seed=PSG_SEED, input_path=INPUT_PATH,
                stimulus=STIMULUS, stim_files={}):
    if not os.path.isdir(input_path):
        os.makedirs(input_path)
        print("The new input directory is created!")

    # Get nodes in pandas dataframe
    nodes = util.load_nodes_from_config("config.json")
    pop_names = ['CP', 'CS', 'FSI', 'LTS']
    Cortex_nodes = get_populations(nodes['cortex'], pop_names, only_id=True)

    # Determines node ids for baseline input
    if 'baseline' in stimulus:
        split_idx = np.cumsum([len(n) for n in Cortex_nodes.values()])
        Base_nodes = np.split(df2node_id(nodes['baseline']), split_idx)
        Base_nodes = dict(zip(pop_names, [n.tolist() for n in Base_nodes[:-1]]))
        input_pairs_to_file(os.path.join(input_path, "Baseline_ids.csv"),
                            Base_nodes.values(), Cortex_nodes.values())

    # Assign assemblies for PNs
    assembly_id_file = os.path.join(input_path, "Assembly_ids.csv")
    if n_assemblies > 0:
        Thal_nodes = df2node_id(nodes['thalamus'])
        Thal_assy, PN_assy = get_assembly(Thal_nodes, Cortex_nodes, n_assemblies)
        input_pairs_to_file(assembly_id_file, Thal_assy, PN_assy)
    elif 'baseline' in stimulus:
        raise ValueError("Use nonzero `n_assemblies` when building baseline")
    else:
        Thal_assy, _ = input_pairs_from_file(assembly_id_file)
        n_assemblies = len(Thal_assy)

    print("Building all input spike trains...")
    start_timer = time.perf_counter()

    # Poisson input mean firing rates
    PN_baseline_fr = 20.0  # Hz. Firing rate for baseline input to PNs
    ITN_baseline_fr = 20.0  # Hz. Firing rate for baseline input to ITNs
    Thal_burst_fr = 50.0 if burst_fr is None else burst_fr  # Hz. for thalamus burst input
    Thal_const_fr = 10.0 if burst_fr is None else burst_fr  # Hz. for thalamus constant input

    std_stim_params = {'n_assemblies': n_assemblies}  # standard stimulus parameters
    # Baseline input
    if 'baseline' in stimulus:
        std_stim_params['baseline'] = dict(t_stop=t_stop,
            PN_firing_rate=PN_baseline_fr, ITN_firing_rate=ITN_baseline_fr)
        fr_params = get_std_param(std_stim_params, 'baseline')
        psg = PoissonSpikeGenerator(population='baseline', seed=psg_seed)
        psg = get_psg_from_fr(psg, [Base_nodes['CP'] + Base_nodes['CS'],
            Base_nodes['FSI'] + Base_nodes['LTS']], fr_params)
        psg.to_sonata(os.path.join(input_path, "baseline.h5"))

    # Constant thalamus input
    if 'const' in stimulus:
        std_stim_params['const'] = dict(t_stop=t_stop, firing_rate=Thal_const_fr)
        fr_params = get_std_param(std_stim_params, 'const')
        psg = PoissonSpikeGenerator(population='thalamus', seed=psg_seed + 100)
        psg = get_psg_from_fr(psg, Thal_assy, fr_params)
        psg.to_sonata(os.path.join(input_path, "thalamus_const.h5"))
        

    # Short burst thalamus input
    if 'short' in stimulus:
        std_stim_params['short'] = dict(firing_rate=Thal_burst_fr,
            on_time=on_time, off_time=off_time, t_start=t_start, t_stop=t_stop)
        fr_params = get_std_param(std_stim_params, 'short')
        psg = PoissonSpikeGenerator(population='thalamus', seed=psg_seed + 100)
        psg = get_psg_from_fr(psg, Thal_assy, fr_params)
        psg.to_sonata(os.path.join(input_path, "thalamus_short.h5"))


    # Long burst thalamus input
    if 'long' in stimulus:
        std_stim_params['long'] = dict(firing_rate=Thal_burst_fr,
            on_time=on_time, off_time=off_time, t_start=t_start, t_stop=t_stop)
        fr_params = get_std_param(std_stim_params, 'long')
        psg = PoissonSpikeGenerator(population='thalamus', seed=psg_seed + 100)
        psg = get_psg_from_fr(psg, Thal_assy, fr_params)
        psg.to_sonata(os.path.join(input_path, "thalamus_long.h5"))

    write_std_stim_file(stim_params=std_stim_params, input_path=input_path)

    # Ramping thalamus input
    if 'ramp' in stimulus:
        stim_setting, stim_file = load_stim_file(input_path=input_path,
            stim_file=stim_files.get('ramp', None), file_name='thalamus_ramp')
        fr_params, stim_setting = get_ramp_param(stim_setting=stim_setting,
            firing_rate=1.5 * Thal_burst_fr)
        assy_idx = stim_setting['setting']['assembly_index']
        psg = PoissonSpikeGenerator(population='thalamus', seed=psg_seed + 100)
        psg = get_psg_from_fr(psg, [Thal_assy[i] for i in assy_idx], fr_params)
        psg.to_sonata(stim_file.replace('.json', '.h5'))
        with open(stim_file, 'w') as f:
            json.dump(stim_setting, f, indent=2)

    # Joining thalamus input
    if 'join' in stimulus:
        stim_setting, stim_file = load_stim_file(input_path=input_path,
            stim_file=stim_files.get('join', None), file_name='thalamus_join')
        fr_params, stim_setting, split_ids = get_join_param(
            stim_setting=stim_setting, size_assemblies=[*map(len, Thal_assy)],
            firing_rate=1.5 * Thal_burst_fr, seed=psg_seed + 200)
        assy_idx = stim_setting['setting']['assembly_index']
        psg = PoissonSpikeGenerator(population='thalamus', seed=psg_seed + 100)
        psg = get_psg_from_fr(psg, split_join_assemblies(
            [Thal_assy[i] for i in assy_idx], split_ids), fr_params)
        psg.to_sonata(stim_file.replace('.json', '.h5'))
        with open(stim_file, 'w') as f:
            json.dump(stim_setting, f, indent=2)

    # Fading thalamus input
    if 'fade' in stimulus:
        stim_setting, stim_file = load_stim_file(input_path=input_path,
            stim_file=stim_files.get('fade', None), file_name='thalamus_fade')
        fr_params, stim_setting = get_fade_param(stim_setting=stim_setting,
            firing_rate=1.5 * Thal_burst_fr)
        assy_idx = stim_setting['setting']['assembly_index']
        psg = PoissonSpikeGenerator(population='thalamus', seed=psg_seed + 100)
        psg = get_psg_from_fr(psg, [Thal_assy[i] for i in assy_idx], fr_params)
        psg.to_sonata(stim_file.replace('.json', '.h5'))
        with open(stim_file, 'w') as f:
            json.dump(stim_setting, f, indent=2)

    print("Core cells: %.3f sec" % (time.perf_counter() - start_timer))

    # These inputs are for the baseline firing rates of the cells in the shell.
    if 'shell' in nodes and 'baseline' in stimulus:
        start_timer = time.perf_counter()

        # Generate Poisson spike trains for shell cells
        psg = PoissonSpikeGenerator(population='shell', seed=psg_seed + 1000)
        shell_nodes = get_populations(nodes['shell'], pop_names, only_id=True)

        # Select effective nodes in shell that only has connections to core
        edge_paths = util.load_config("config.json")['networks']['edges']
        _, shell_edges = util.load_edges(**next(
            path for path in edge_paths if 'shell_cortex' in path['edges_file']))
        effective_shell = set(shell_edges['source_node_id'])

        print("Proportion of effective cells in shell.")
        fr_list = []
        for p, node_ids in shell_nodes.items():
            effective_ids = [x for x in node_ids if x in effective_shell]
            ratio = len(effective_ids) / len(node_ids)
            print("%.1f%% effective %s." % (100 * ratio, p))

            fr = SHELL_FR.loc[p]
            if SHELL_CONSTANT_FR:
                # Constant mean firing rate for all cells
                psg.add(node_ids=effective_ids,
                        firing_rate=fr['mean'], times=(0, t_stop))
            else:
                # Lognormal distributed mean firing rate
                fr_list.append([effective_ids, psg_lognormal_fr(psg, effective_ids,
                    mean=fr['mean'], stdev=fr['stdev'], times=(0, t_stop))])

        SHELL_FR.to_csv(os.path.join(input_path, "Shell_FR_stats.csv"))
        if not SHELL_CONSTANT_FR:
            fr_list = {k: np.concatenate(v) for k, v in
                       zip(['node_id','firing_rate'], zip(*fr_list))}
            fr_list = pd.DataFrame(fr_list).set_index('node_id')
            fr_list.to_csv(os.path.join(input_path, "Lognormal_FR.csv"))

        psg.to_sonata(os.path.join(input_path, "shell.h5"))
        print("Shell cells: %.3f sec" % (time.perf_counter() - start_timer))

    write_seeds_file(psg_seed=psg_seed, net_seed=NET_SEED, stimulus=stimulus,
                     input_path=input_path, seeds_file_name='random_seeds')
    print("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-stop', '--t_stop', type=float,
                        nargs='?', default=T_STOP,
                        help="Simulation stop time", metavar='t_stop')
    parser.add_argument('-start', '--t_start', type=float,
                        nargs='?', default=T_START,
                        help="Simulation start period", metavar='t_start')
    parser.add_argument('-n', '--n_assemblies', type=int,
                        nargs='?', default=N_ASSEMBLIES,
                        help="Number of assemblies", metavar='# Assemblies')
    parser.add_argument('-fr', '--burst_fr', type=float,
                        nargs='?', default=None,
                        help="Thalamus burst input firing rate", metavar='Firing Rate')
    parser.add_argument('-net', '--net_seed', type=int,
                        nargs='?', default=NET_SEED,
                        help="Network random seed", metavar='Network Seed')
    parser.add_argument('-psg', '--psg_seed', type=int,
                        nargs='?', default=PSG_SEED,
                        help="Poisson generator seed", metavar='PSG Seed')
    parser.add_argument('-path', '--input_path', type=str,
                        nargs='?', default=INPUT_PATH,
                        help="Input path", metavar='Input Path')
    parser.add_argument('-s', '--stimulus', type=str,
                        nargs="*", default=STIMULUS,
                        help="List of stimulus types", metavar='Stimulus')
    parser.add_argument('-f', '--stim_files', type=str,
                        nargs="*", default=[], metavar='Stimulus Files',
                        help="Key value pairs of stimulus type and file path, "
                        "e.g. stim1 file1 stim2 file2")
    args = parser.parse_args()

    stimulus = args.stimulus
    stim_files = args.stim_files
    if len(stim_files) % 2:
        if len(stim_files) == 1 and len(stimulus) == 1:
            stim_files = {stimulus[0]: stim_files[0]}
        else:
            raise ValueError("Number of keys and values in stim_files should match")
    else:
        stim_files = dict(zip(stim_files[::2], stim_files[1::2]))

    NET_SEED = args.net_seed
    rng = np.random.default_rng(NET_SEED)

    build_input(t_stop=args.t_stop, t_start=args.t_start,
                n_assemblies=args.n_assemblies, burst_fr=args.burst_fr,
                psg_seed=args.psg_seed, input_path=args.input_path,
                stimulus=stimulus, stim_files=stim_files)
