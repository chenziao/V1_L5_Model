import numpy as np
import pandas as pd
import xarray as xr
import json
import h5py
import os
import copy


ROOT_DIR_NAME = 'V1_L5_Model'

STIMULUS_CONFIG = {
    'baseline': 'config_baseline.json',
    'short': 'config_short.json',
    'long': 'config_long.json',
    'const': 'config_const.json',
    'ramp': 'config.json',
    'join': 'config.json',
    'fade': 'config.json',
    'else': 'config.json'
}


class ConfigHelper(object):
    def __init__(self, config_file, root_dir_name=ROOT_DIR_NAME):
        self.config_file = os.path.abspath(config_file)
        self.root_dir_name = root_dir_name
        with open(self.config_file, 'r') as f:
            self.config = json.load(f)
        self.root_dir = self.get_root_dir(self.config['config_dir'])
        self.root_cwd = self.get_root_dir(os.getcwd())
        self.get_config_cwd()

    def get_root_dir(self, abs_path):
        """Find the directory in absolute path that matches root_dir_name"""
        while True:
            path = abs_path
            abs_path, tail = os.path.split(abs_path)
            if tail == '':
                raise NotADirectoryError("Root directory not found")
            if self.root_dir_name in tail:
                break
        return os.path.normpath(path)

    def get_attr(self, *key_chain):
        """Get config attributes from a chain of keys"""
        value = self.config_cwd
        for key in key_chain:
            value = value.get(key)
            if value is None:
                break
        return value

    def get_file_cwd(self, file_path):
        """Get file in current working directory"""
        if isinstance(file_path, str):
            path = os.path.normpath(file_path)
            try:
                root = os.path.commonpath((path, self.root_dir))
            except:
                pass
            else:
                if self.root_dir == root:
                    rel_path = os.path.relpath(path, start=self.root_dir)
                    file_path = os.path.normpath(os.path.join(self.root_cwd, rel_path))
        return file_path

    def get_config_cwd(self):
        self.config_cwd = copy.deepcopy(self.config)
        map_json_inplace(self.config_cwd, self.get_file_cwd)


def stimulus_type_from_trial_name(trial_name):
    stim_type = next(s for s in trial_name.split('_') if s in STIMULUS_CONFIG)
    return stim_type, STIMULUS_CONFIG[stim_type]


def get_trial_info(TRIAL_PATH):
    _, TRIAL_NAME = os.path.split(TRIAL_PATH)
    stimulus_type, config = stimulus_type_from_trial_name(TRIAL_NAME)
    isbaseline = stimulus_type == 'baseline' or stimulus_type == 'const'
    isstandard = isbaseline or stimulus_type == 'short' or stimulus_type == 'long'

    result_config_file = os.path.join(TRIAL_PATH, 'config_no_STP.json'
                                      if 'no_STP' in TRIAL_NAME else config)
    if not os.path.isfile(result_config_file):
        result_config_file = os.path.join(os.path.split(result_config_file)[0],
                                          STIMULUS_CONFIG['else'])

    config_hp = ConfigHelper(result_config_file)
    t_stop = config_hp.get_attr('run', 'tstop') / 1000

    INPUT_PATH, _ = os.path.split(config_hp.get_attr('inputs', 'baseline_spikes', 'input_file'))
    STIM_FILE = config_hp.get_attr('inputs', 'thalamus_spikes', 'input_file')
    NODE_FILES = config_hp.get_attr('networks', 'nodes')
    SPIKE_FILE = os.path.join(TRIAL_PATH, os.path.split(
        config_hp.get_attr('output', 'spikes_file'))[1])

    stim_file = 'standard_stimulus' if isstandard \
        else os.path.splitext(os.path.split(STIM_FILE)[1])[0]
    with open(os.path.join(INPUT_PATH, stim_file + '.json')) as f:
        stim_setting = json.load(f)
    stim_params = stim_setting[stimulus_type if isstandard else 'stim_params']

    stim_type = (stimulus_type, isbaseline, isstandard)
    paths = (INPUT_PATH, NODE_FILES, SPIKE_FILE)
    stim_info = (t_stop, stim_setting, stim_params)
    return stim_type, paths, stim_info, config_hp


def map_json_inplace(val, func, obj=None, key=None):
    if isinstance(val, dict):
        for k, v in val.items():
            map_json_inplace(v, func, obj=val, key=k)
    elif isinstance(val, list):
        for k, v in enumerate(val):
            map_json_inplace(v, func, obj=val, key=k)
    else:
        obj[key] = func(obj[key])


def load_spikes_to_df(spike_file, network_name):
    with h5py.File(spike_file) as f:
        spikes_df = pd.DataFrame({
            'node_ids': f['spikes'][network_name]['node_ids'],
            'timestamps': f['spikes'][network_name]['timestamps']
        })
        spikes_df.sort_values(by='timestamps', inplace=True, ignore_index=True)
    return spikes_df


def load_ecp_to_xarray(ecp_file, demean=False):
    with h5py.File(ecp_file, 'r') as f:
        ecp = xr.DataArray(
            f['ecp']['data'][()].T,
            coords = dict(
                channel_id = f['ecp']['channel_id'][()],
                time = np.arange(*f['ecp']['time']) # ms
            ),
            attrs = dict(
                fs = 1000 / f['ecp']['time'][2] # Hz
            )
        )
    if demean:
        ecp -= ecp.mean(dim='time')
    return ecp
