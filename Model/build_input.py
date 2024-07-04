#!/usr/bin/env

import os
import csv
import json
import time
import argparse

import numpy as np
import pandas as pd
from functools import partial
from bmtool.util import util
from bmtk.utils.reports.spike_trains import PoissonSpikeGenerator
from connectors import num_prop


INPUT_PATH = "./input"
STIMULUS = ['baseline', 'short', 'long', 'shell']

N_ASSEMBLIES = 9  # number of assemblies
NET_SEED = 123  # random seed for network r.v.'s (e.g. assemblies, firing rate)
PSG_SEED = 1  # poisson spike generator random seed for different trials
# Warning: Using PoissonSpikeGenerator(seed=0) may not set random seed correctly.

T_STOP = 28.  # sec. Simulation time
T_START = 1.0  # sec. Time to start burst input
t_start = T_START  # to be imported to other scipts
on_time = 1.0  # sec. Burst input duration
off_time = 0.5  # sec. Silence duration
off_time_expr = 1.0  # sec. Silence duration for experiments (longer for reset)
n_cycles_expr = 10  # number of cycles for experiments

SHELL_FR = {
    'CP': (1.8, 1.4),
    'CS': (1.4, 1.1),
    'FSI': (7.5, 6.0),
    'LTS': (3.5, 4.0)
}  # firing rate of shell neurons (mean, stdev)
SHELL_FR = pd.DataFrame.from_dict(
    SHELL_FR, orient='index', columns=('mean', 'stdev')).rename_axis('pop_name')
SHELL_CONSTANT_FR = False  # whether use constant firing rate for shell neurons


def get_rng(seed=NET_SEED, seed_offset=0):
    return np.random.default_rng(seed + seed_offset)

default_rng = get_rng()


def num_prop(ratio, N):
    """Calculate numbers of total N in proportion to ratio"""
    ratio = np.asarray(ratio)
    p = np.cumsum(np.insert(ratio.ravel(), 0, 0))  # cumulative proportion
    return np.diff(np.round(N / p[-1] * p).astype(int)).reshape(ratio.shape)


def lognormal(mean, stdev, size=None, rng=default_rng):
    """Generate random values from lognormal given mean and stdev"""
    sigma2 = np.log((stdev / mean) ** 2 + 1)
    mu = np.log(mean) - sigma2 / 2
    sigma = sigma2 ** 0.5
    return rng.lognormal(mu, sigma, size)


def psg_lognormal_fr(psg, node_ids, mean, stdev, times, rng=default_rng):
    """Generate lognormal distributed firing rate for each node independently.
    Then add the firing rate of each node to the given PoissonSpikeGenerator.
    """
    firing_rates = lognormal(mean, stdev, len(node_ids), rng=rng)
    for node_id, fr in zip(node_ids, firing_rates):
        psg.add(node_ids=node_id, firing_rate=fr, times=times)
    return firing_rates


def df2node_id(df):
    """Get node ids from a node dataframe into a list"""
    return df.index.tolist()


def get_pop(node_df, value, key='pop_name'):
    """Get dataframe of nodes matching a specific property from nodes dataframe
    key, value: key-value pair of a property. Default property: population name 
    """
    return node_df.loc[node_df[key] == value]


def get_pop_id(node_df, value, key='pop_name'):
    """Get ids of nodes matching a specific property from nodes dataframe"""
    return df2node_id(get_pop(node_df, value, key=key))


def get_populations(node_df, values, key='pop_name', only_id=False):
    """Get a dictionary of {value: nodes} matching different values of
    a property from nodes dataframe. Default property: population name
    only_id: whether return only node ids or a dataframe of nodes
    """
    func = partial(get_pop_id, key=key) if only_id else partial(get_pop, key=key)
    return {v: func(node_df, v) for v in values}


def assign_assembly(N, n_assemblies, rng=default_rng):
    """Assign N units to n_assemblies.
    Return a list of unit indices in each assembly.
    """
    n_per_assemb = num_prop(np.ones(n_assemblies), N)
    split_idx = np.cumsum(n_per_assemb)[:-1]  # indices at which to split
    assy_idx = rng.permutation(N)  # random shuffle for assemblies
    assy_idx = np.split(assy_idx, split_idx)  # split into assemblies
    assy_idx = [np.sort(idx) for idx in assy_idx]
    return assy_idx


def get_assembly_ids(*pop_nodes, assy_idx=[slice(None)]):
    """Cast node ids into a list of assemblies given indices in each assembly."""
    pop_assy = []
    for nodes in pop_nodes:
        ids = np.array(nodes)
        pop_assy.append([ids[idx] for idx in assy_idx])
    return pop_assy


def get_assembly(Thal_nodes, PN_nodes, n_assemblies, rng=default_rng):
    """Divide PNs into n_assemblies and return lists of ids in each assembly"""
    num_PN = len(PN_nodes)
    if len(Thal_nodes) != num_PN:
        raise ValueError("Number of thalamus cells don't match number of PNs")

    assy_idx = assign_assembly(num_PN, n_assemblies, rng=rng)
    Thal_assy, PN_assy = get_assembly_ids(Thal_nodes, PN_nodes, assy_idx=assy_idx)
    return Thal_assy, PN_assy


def get_divided_assembly(Thal_nodes, PN_nodes_df, div_assembly, rng=default_rng):
    """Divide PNs assemblies into smaller assemblies.
    div_assembly: If a single number is specified, it is the number of smaller
        assemblies that each of the original assembly will be divided into.
        If specified as a list, it is the sequence of original assembly ids
        from which the smaller assemblies will be taken from.
        E.g., div_assembly=[3] divides each assembly into 3 smaller ones.
        div_assembly=[1, 0, 2, 2, 1] divides assembly 0 into one, 1 into two and
        2 into two smaller assemblies, ordered corresponding to the sequence.
    """
    if len(PN_nodes_df) != len(Thal_nodes):
        raise ValueError("Number of thalamus cells don't match number of PNs")
    assy_ids = PN_nodes_df['assembly_id'].unique()
    assy_ids = np.sort(assy_ids[assy_ids >= 0])
    div_assembly = np.array(div_assembly).ravel()
    if div_assembly.size == 1:
        # divide each original assembly into equal number of smaller ones 
        # small assemblies order is sequentially switching among the orginal ones
        div_assembly = np.tile(assy_ids, div_assembly[0])
    if not set(div_assembly).issubset(assy_ids):
        s = ', '.join(map(str, set(div_assembly) - set(assy_ids)))
        raise ValueError("The assembly id (%s) in `div_assembly` not found "
                         "in the network." % s)
    n_div = {i: np.count_nonzero(div_assembly == i) for i in assy_ids}
    PN_Assy = get_populations(PN_nodes_df, assy_ids, key='assembly_id', only_id=True)
    PN_Assy_div = {i: get_assembly_ids(PN_Assy[i], assy_idx=assign_assembly(
        len(PN_Assy[i]), n_div[i], rng=rng))[0] for i in assy_ids}
    PN_assy = [PN_Assy_div[i].pop(0) for i in div_assembly]
    assy_idx = [PN_nodes_df.index.get_indexer(ids) for ids in PN_assy]
    Thal_assy, = get_assembly_ids(Thal_nodes, assy_idx=assy_idx)
    return Thal_assy, PN_assy, div_assembly


GRID_SIZE = np.array([[-300., 300.], [-300., 300.]]) # um. x, y bounds
GRID_ID = np.array([
    [6, 2, 8],
    [0, 4, 5],
    [3, 7, 1]
])
def get_grid_assembly(Thal_nodes, PN_nodes_df,
                      grid_id=GRID_ID, grid_size=GRID_SIZE):
    """Divide PNs into assemblies based on lateral location (x, y).
    The layer is divided into a 2D grid. Cells in each grid form an assembly.
    grid_id: assembly ids arranged in 2d-array corresponding to grid locations.
        The assemblies are ordered by the ids.
    grid_size: the bounds of the grid area in (x, y) coordinates (um).
    """
    if len(PN_nodes_df) != len(Thal_nodes):
        raise ValueError("Number of thalamus cells don't match number of PNs")
    bins = []
    for i in range(2):
        bins.append(np.linspace(*grid_size[i], grid_id.shape[i] + 1)[1:])
        bins[i][-1] += 1.
    PN_nodes_df['assy_id'] = grid_id[np.digitize(PN_nodes_df['pos_x'], bins[0]),
                                     np.digitize(PN_nodes_df['pos_y'], bins[1])]

    assy_idx = [PN_nodes_df['assy_id'] == i for i in np.sort(grid_id, axis=None)]
    Thal_assy, PN_assy = get_assembly_ids(Thal_nodes, PN_nodes_df.index, assy_idx=assy_idx)
    return Thal_assy, PN_assy, grid_id


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
    if not os.path.isfile(file):
        raise FileNotFoundError("%s has not been created." % file)
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
        ax.set_title('Assembly %d' % i)
    ax.set_xlabel('Time (sec)')
    plt.tight_layout()
    return fig, axs


def get_fr_short(n_assemblies, firing_rate=(0., 0.),
                 on_time=on_time, off_time=off_time,
                 t_start=T_START, t_stop=T_STOP, n_rounds=1):
    """Short burst is delivered to each assembly sequentially within each cycle.
    n_assemblies: number of assemblies
    firing_rate: 2-tuple of firing rate at off and on time, respectively
        Pad zero to the beginning if sequence is shorter than required
    t_start, t_stop: start and stop time of the stimulus cycles
    n_rounds: number of short bursts each assembly receives per cycle.
        It can be a fractional number. In this case, some fraction of assemblies
        will receive one more short burst per cycle than others.
    Return: firing rate traces
    """
    firing_rate = np.asarray(firing_rate).ravel()[:2]
    firing_rate = np.concatenate((np.zeros(2 - firing_rate.size), firing_rate))
    t_cycle, n_cycle = get_stim_cycle(on_time, off_time, t_start, t_stop)

    n_bursts = int(np.ceil(n_rounds * n_assemblies))
    n_rounds = int(np.ceil(n_rounds))
    times = np.empty((n_assemblies, n_rounds * n_cycle * 4 + 2))
    times[:, 0] = 0.
    times[:, -1] = t_stop
    on_times = np.linspace(0, on_time, n_bursts + 1)
    fr = np.append(np.tile(firing_rate, n_rounds * n_cycle), firing_rate[0])
    fr = np.tile(np.repeat(fr, 2), (n_assemblies, 1))
    for j in range(n_cycle):
        ts = t_start + t_cycle * j + on_times
        for k in range(n_rounds):
            t = (j * n_rounds + k) * 4 + 1
            for i in range(n_assemblies):
                tt = k * n_assemblies + i
                if tt < n_bursts:
                    times[i, t:t + 4] = np.repeat(ts[tt:tt + 2], 2)
                else:
                    times[i, t:t + 4] = ts[-1]
                    fr[i, t:t + 4] = firing_rate[0]

    params = [dict(firing_rate=fr[i], times=times[i]) for i in range(n_assemblies)]
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
        rng = get_rng(seed=NET_SEED, seed_offset=seed)  # shuffle ids in each assembly
    for n in size_assemblies:
        assy_idx = np.arange(n) if seed is None else rng.permutation(n)  
        n_per_step = num_prop(ratio, n * high_portion)  # split into steps
        split_idx = np.cumsum(n_per_step)  # indices at which to split
        split_ids.append(np.split(assy_idx, split_idx)[:-1])
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
    p = stim_setting.get(stimulus).copy()
    n_assemblies = stim_setting.get('n_assemblies', N_ASSEMBLIES)
    if 'baseline' in stimulus:
        times = (0, p.pop('t_stop', T_STOP))
        fr_params = [{'firing_rate': fr, 'times': times} for fr in p.values()]
    elif 'const' in stimulus:
        t_stop = p.get('t_stop', T_STOP)
        fr_params = get_fr_long(n_assemblies, [p['firing_rate']] * 2,
                                on_time=t_stop, off_time=0., t_stop=t_stop)
    elif 'short' in stimulus:
        fr_params = get_fr_short(n_assemblies, **p)
    elif 'long' in stimulus:
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
        stim_file = new_file_name(input_path, file_name, '.json')
        stim_setting = {}
        print("Stimulus file for %s not specified. "
              "Using default settings." % file_name)
    else:
        _, ext = os.path.splitext(stim_file)
        if ext != '.json':
            stim_file += '.json'
        loading_stim_file = stim_file if os.path.isfile(stim_file) else None
        file_name = os.path.split(stim_file)[1]
        stim_file = os.path.join(input_path, file_name)
        if loading_stim_file is None and os.path.isfile(stim_file):
            loading_stim_file = stim_file
        if loading_stim_file is None:
            stim_setting = None
            print("Stimulus file %s not found." % file_name)
        else:
            with open(loading_stim_file, 'r') as f:
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
        seed['stimulus'].extend(stimulus_new)
        overwrite = bool(stimulus_new)
    else:
        seed = dict(net_seed=net_seed, psg_seed=psg_seed, stimulus=stimulus)
        seeds.append(seed)
        overwrite = True
    if overwrite:
        with open(seeds_file, 'w') as f:
            json.dump(seeds, f, indent=2)


def build_input(t_stop=T_STOP, t_start=T_START, n_assemblies=N_ASSEMBLIES,
                div_assembly=None, grid_assembly=False,
                burst_fr=None, net_seed=NET_SEED, psg_seed=PSG_SEED,
                input_path=INPUT_PATH, stimulus=STIMULUS, stim_files={}):
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
        PN_nodes = Cortex_nodes['CP'] + Cortex_nodes['CS']
        if div_assembly is not None:
            rng = get_rng(seed=net_seed, seed_offset=100)
            Thal_assy, PN_assy, div_assembly = get_divided_assembly(
                Thal_nodes, nodes['cortex'].loc[PN_nodes], div_assembly, rng=rng)
            n_assemblies = len(Thal_assy)
            div_assembly_df = pd.DataFrame(enumerate(div_assembly), columns=[
                'division_id', 'assembly_id']).set_index('division_id')
            div_assembly_df.to_csv(os.path.join(input_path, "Division_ids.csv"))
        elif grid_assembly:
            Thal_assy, PN_assy, grid_id = get_grid_assembly(
                Thal_nodes, nodes['cortex'].loc[PN_nodes])
            n_assemblies = len(Thal_assy)
            gird_id_file = os.path.join(input_path, "Grid_ids.csv")
            np.savetxt(gird_id_file, grid_id, fmt='%d', delimiter=",")
        else:
            rng = get_rng(seed=net_seed, seed_offset=100)
            Thal_assy, PN_assy = get_assembly(
                Thal_nodes, PN_nodes, n_assemblies, rng=rng)
        input_pairs_to_file(assembly_id_file, Thal_assy, PN_assy)
    else:
        try:
            Thal_assy, _ = input_pairs_from_file(assembly_id_file)
        except FileNotFoundError as e:
            raise FileNotFoundError("Use nonzero `n_assemblies`") from e
        n_assemblies = len(Thal_assy)

    print("Building all input spike trains...")
    start_timer = time.perf_counter()

    # Poisson input mean firing rates
    PN_baseline_fr = 20.0  # Hz. Firing rate for baseline input to PNs
    ITN_baseline_fr = 20.0  # Hz. Firing rate for baseline input to ITNs
    FSI_baseline_fr = None  # 200 Hz. If not None, use modified fr for FSI
    LTS_baseline_fr = 60.0  # 60 Hz. If not None, use modified fr for LTS
    ITN_baseline_burst = True  # whether use burst input like short/long or constant
    Thal_burst_fr = 50.0 if burst_fr is None else burst_fr  # Hz. for thalamus burst input
    Thal_const_fr = 10.0 if burst_fr is None else burst_fr  # Hz. for thalamus constant input

    def PSG(population='thalamus', seed_offset=100):
        seed = psg_seed + seed_offset
        print("Using random seed %g in %s population." % (seed, population))
        return PoissonSpikeGenerator(population=population, seed=seed)

    std_stim_params = {'n_assemblies': n_assemblies}  # standard stimulus parameters
    for stim in stimulus:
        # Baseline input
        if stim == 'baseline':
            stim_setting = dict(t_stop=t_stop, PN_firing_rate=PN_baseline_fr)
            if FSI_baseline_fr is None and LTS_baseline_fr is None:
                # normal baseline input
                stim_setting['ITN_firing_rate'] = ITN_baseline_fr
                ITN_nodes = [Base_nodes['FSI'] + Base_nodes['LTS']]
                ITN_baseline_burst = False
            else:
                # supply pseudo stimulus input to ITNs as baseline (for PING validation)
                stim_setting['FSI_firing_rate'] = ITN_baseline_fr \
                    if FSI_baseline_fr is None else FSI_baseline_fr
                stim_setting['LTS_firing_rate'] = ITN_baseline_fr \
                    if LTS_baseline_fr is None else LTS_baseline_fr
                ITN_nodes = [Base_nodes['FSI'], Base_nodes['LTS']]
            std_stim_params['baseline'] = stim_setting
            fr_params = get_std_param(std_stim_params, 'baseline')
            if ITN_baseline_burst:
                stim_setting = dict(on_time=on_time, off_time=off_time,
                                   t_start=t_start, t_stop=t_stop)
                for i, fr in enumerate([FSI_baseline_fr, LTS_baseline_fr]):
                    if fr is not None:
                        fr = [ITN_baseline_fr, fr]
                        fr_params[i + 1] = get_fr_long(1, fr, **stim_setting)[0]
                std_stim_params['baseline'].update(stim_setting)
            psg = PSG(population='baseline', seed_offset=0)
            psg = get_psg_from_fr(psg, [Base_nodes['CP'] + Base_nodes['CS']] \
                + ITN_nodes, fr_params)
            psg.to_sonata(os.path.join(input_path, "baseline.h5"))
            continue

        if stim == 'shell':
            continue

        if any(s in stim for s in ('short', 'long', 'const')):
            if 'const' in stim:
                # Constant thalamus input
                std_stim_params[stim] = dict(t_stop=t_stop, firing_rate=Thal_const_fr)
            else:
                # Short/Long burst thalamus input
                std_stim_params[stim] = dict(firing_rate=Thal_burst_fr,
                    on_time=on_time, off_time=off_time, t_start=t_start, t_stop=t_stop)
                if stim == 'short':
                    std_stim_params[stim]['n_rounds'] = 1
            fr_params = get_std_param(std_stim_params, stim)
            psg = get_psg_from_fr(PSG(), Thal_assy, fr_params)
            psg.to_sonata(os.path.join(input_path, "thalamus_" + stim + ".h5"))
            continue

        # Special stimulus types
        stim_setting, stim_file = load_stim_file(input_path=input_path,
            stim_file=stim_files.get(stim, None), file_name='thalamus_' + stim)
        if stim_setting is None:
            print("Skiping stimulus %s." % stim)
            continue

        if 'ramp' in stim:
            # Ramping thalamus input
            fr_params, stim_setting = get_ramp_param(stim_setting=stim_setting,
                firing_rate=1.5 * Thal_burst_fr)
            assy_idx = stim_setting['setting']['assembly_index']
            psg = get_psg_from_fr(PSG(), [Thal_assy[i] for i in assy_idx], fr_params)
        elif 'join' in stim:
            # Joining thalamus input
            fr_params, stim_setting, split_ids = get_join_param(
                stim_setting=stim_setting, size_assemblies=[*map(len, Thal_assy)],
                firing_rate=1.5 * Thal_burst_fr, seed=psg_seed + 200)
            assy_idx = stim_setting['setting']['assembly_index']
            psg = get_psg_from_fr(PSG(), split_join_assemblies(
                [Thal_assy[i] for i in assy_idx], split_ids), fr_params)
        elif 'fade' in stim:
            # Fading thalamus input
            fr_params, stim_setting = get_fade_param(stim_setting=stim_setting,
                firing_rate=1.5 * Thal_burst_fr)
            assy_idx = stim_setting['setting']['assembly_index']
            psg = get_psg_from_fr(PSG(), [Thal_assy[i] for i in assy_idx], fr_params)
        else:
            print("Stimulus %s not defined. Skipping." % stim)
            continue
        psg.to_sonata(stim_file.replace('.json', '.h5'))
        with open(stim_file, 'w') as f:
            json.dump(stim_setting, f, indent=2)

    write_std_stim_file(stim_params=std_stim_params, input_path=input_path)
    print("Core cells: %.3f sec" % (time.perf_counter() - start_timer))

    # These inputs are for the baseline firing rates of the cells in the shell.
    shell = 'shell' in stimulus and 'shell' in nodes
    if shell:
        start_timer = time.perf_counter()

        # Generate Poisson spike trains for shell cells
        psg = PSG(population='shell', seed_offset=1000)
        shell_nodes = get_populations(nodes['shell'], pop_names, only_id=True)

        # Select effective nodes in shell that only has connections to core
        edge_paths = util.load_config("config.json")['networks']['edges']
        _, shell_edges = util.load_edges(**next(
            path for path in edge_paths if 'shell_cortex' in path['edges_file']))
        effective_shell = set(shell_edges['source_node_id'])

        fr_list = []
        if not SHELL_CONSTANT_FR:
            rng = get_rng(seed=net_seed)
        print("Proportion of effective cells in shell.")
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
                    mean=fr['mean'], stdev=fr['stdev'], times=(0, t_stop), rng=rng)])

        SHELL_FR.to_csv(os.path.join(input_path, "Shell_FR_stats.csv"))
        if not SHELL_CONSTANT_FR:
            fr_list = {k: np.concatenate(v) for k, v in
                       zip(['node_id','firing_rate'], zip(*fr_list))}
            fr_list = pd.DataFrame(fr_list).set_index('node_id')
            fr_list.to_csv(os.path.join(input_path, "Lognormal_FR.csv"))

        psg.to_sonata(os.path.join(input_path, "shell.h5"))
        print("Shell cells: %.3f sec" % (time.perf_counter() - start_timer))
    else:
        if 'shell' in stimulus:
            stimulus.remove('shell')

    write_seeds_file(psg_seed=psg_seed, net_seed=net_seed, stimulus=stimulus,
                     input_path=input_path, seeds_file_name='random_seeds')
    print("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-stop', '--t_stop', type=float,
                        nargs='?', default=T_STOP, metavar='t_stop',
                        help="Simulation stop time")
    parser.add_argument('-start', '--t_start', type=float,
                        nargs='?', default=T_START, metavar='t_start',
                        help="Simulation start period")
    parser.add_argument('-n', '--n_assemblies', type=int,
                        nargs='?', default=N_ASSEMBLIES, metavar='# Assemblies',
                        help="Number of assemblies to randomly assign. "
                        "Set to 0 to load assemblies that have already been assigned.")
    parser.add_argument('-div', '--div_assembly', type=int,
                        nargs="*", default=None, metavar='Divide assemblies',
                        help="Number of smaller assemblies that each of the "
                        "original assembly will be randomly divided into. "
                        "Or a sequence of original assembly ids from which "
                        "the smaller assemblies will be taken from.")
    parser.add_argument('-grid', '--grid_assembly', action='store_true',
                        help="Use spatial grids to assign assemblies")
    parser.add_argument('-fr', '--burst_fr', type=float,
                        nargs='?', default=None, metavar='Firing Rate',
                        help="Thalamus burst input firing rate")
    parser.add_argument('-net', '--net_seed', type=int,
                        nargs='?', default=NET_SEED, metavar='Network Seed',
                        help="Network random seed")
    parser.add_argument('-psg', '--psg_seed', type=int,
                        nargs='?', default=PSG_SEED, metavar='PSG Seed',
                        help="Poisson generator seed")
    parser.add_argument('-path', '--input_path', type=str,
                        nargs='?', default=INPUT_PATH, metavar='Input Path',
                        help="Input path")
    parser.add_argument('-s', '--stimulus', type=str,
                        nargs="*", default=STIMULUS, metavar='Stimulus',
                        help="List of stimulus types. List can be empty.")
    parser.add_argument('-sf', '--stim_files', type=str,
                        nargs="*", default=[], metavar='Stimulus Files',
                        help="List of stimulus file names/paths. "
                        "Inferred stimulus types are added to the stimulus list. "
                        "E.g., -sf ramp1 ramp2 ./input/join.json")
    args = parser.parse_args()

    NET_SEED = args.net_seed
    stimulus = []
    for s in args.stimulus:
        if s not in stimulus:  # no duplicates
            stimulus.append(s)
    stim_files = {}
    for s_ in args.stim_files:
        s = os.path.split(s_)[1].removesuffix('.json')
        if s not in stim_files:  # no duplicates
            stim_files[s] = s_
    stimulus.extend(s for s in stim_files if s not in stimulus)

    build_input(t_stop=args.t_stop, t_start=args.t_start, n_assemblies=args.n_assemblies,
                div_assembly=args.div_assembly, grid_assembly = args.grid_assembly, 
                burst_fr=args.burst_fr, net_seed=args.net_seed, psg_seed=args.psg_seed,
                input_path=args.input_path, stimulus=stimulus, stim_files=stim_files)
