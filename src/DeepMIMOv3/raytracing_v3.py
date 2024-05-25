# -*- coding: utf-8 -*-
"""
DeepMIMOv3 Python Implementation

Description: Read scenario files

Authors: Umut Demirhan, Ahmed Alkhateeb
Date: 12/10/2021
"""

import os
import numpy as np
from tqdm import tqdm
import DeepMIMOv3.consts as c
from DeepMIMOv3.utils import safe_print, PathVerifier, dbm2pow
import scipy.io
import glob

def read_raytracing(bs_id, params, user=True):
    
    params[c.PARAMSET_SCENARIO_PARAMS] = load_scenario_params(params[c.PARAMSET_SCENARIO_PARAMS_PATH])
    
    if user:
        generation_idx = params[c.PARAMSET_ACTIVE_UE] # Active User IDX
    else:
        generation_idx = params[c.PARAMSET_ACTIVE_BS]-1 # Active BS IDX
        
    ray_data, rx_locs, tx_loc = load_ray_data(generation_idx, bs_id, params, user)
    
    if params['scenario_params']['dual_polar_available'] and params['enable_dual_polar']:
        ray_data_out = {}
        for i, polar_str in enumerate(['VV', 'VH', 'HH', 'HV']):
            ray_data_out[polar_str] = extract_data_from_ray(ray_data[i], rx_locs, params)
    else:
        ray_data_out = extract_data_from_ray(ray_data, rx_locs, params)
    
    return ray_data_out, tx_loc

# Extracts the information to a dictionary from the loaded data file
# for the users given in IDs, and with maximum number of paths
def extract_data_from_ray(ray_data, rx_locs, params):
    
    path_verifier = PathVerifier(params)
    
    num_channels = ray_data.shape[0]
    # Generate empty user array of dictionaries
    path_dict = {c.OUT_PATH_NUM: 0,
                 c.OUT_PATH_DOD_PHI: [],
                 c.OUT_PATH_DOD_THETA: [],
                 c.OUT_PATH_DOA_PHI: [],
                 c.OUT_PATH_DOA_THETA: [],
                 c.OUT_PATH_PHASE: [],
                 c.OUT_PATH_TOA: [],
                 c.OUT_PATH_RX_POW: [],
                 c.OUT_PATH_LOS: []
                 }
    if params[c.PARAMSET_SCENARIO_PARAMS][c.LOAD_FILE_SP_DOPPLER]:
        path_dict[c.OUT_PATH_DOP_VEL] = []
        path_dict[c.OUT_PATH_DOP_ACC] = []
    
    data = {c.OUT_PATH: np.array([dict(path_dict) for j in range(num_channels)]),
            c.OUT_LOS : np.zeros(num_channels, dtype=int)
            }
    if num_channels>0:
        data[c.OUT_LOC] = rx_locs[:, :3]
        data[c.OUT_DIST] = rx_locs[:, 3]
        data[c.OUT_PL] = rx_locs[:, 4]
    else:
        data[c.OUT_LOC] = np.zeros((num_channels, 3), dtype=float)
        data[c.OUT_DIST] = np.zeros(num_channels, dtype=float)
        data[c.OUT_PL] = np.zeros(num_channels, dtype=float)
    
    for user in tqdm(range(num_channels), desc='Reading ray-tracing'):
        num_paths_available = int(ray_data[user][0].shape[1])
        if num_paths_available>0:
            data[c.OUT_PATH][user] = load_variables(ray_data[user][0], path_verifier, params)
        
    path_verifier.notify()
    
    return data

# Split variables into a dictionary
def load_variables(path_params, path_verifier, params):
    num_max_paths = params[c.PARAMSET_NUM_PATHS]
    user_data = dict()
    user_data[c.OUT_PATH_NUM] = min(num_max_paths, path_params.shape[1])
    user_data[c.OUT_PATH_PHASE] = path_params[0, :num_max_paths]
    user_data[c.OUT_PATH_TOA] = path_params[1, :num_max_paths]
    user_data[c.OUT_PATH_RX_POW] = dbm2pow(path_params[2, :num_max_paths] + 30 - params[c.PARAMSET_SCENARIO_PARAMS][c.PARAMSET_SCENARIO_PARAMS_TX_POW])
    user_data[c.OUT_PATH_DOA_PHI] = path_params[3, :num_max_paths]
    user_data[c.OUT_PATH_DOA_THETA] = path_params[4, :num_max_paths]
    user_data[c.OUT_PATH_DOD_PHI] = path_params[5, :num_max_paths]
    user_data[c.OUT_PATH_DOD_THETA] = path_params[6, :num_max_paths]
    user_data[c.OUT_PATH_LOS] =  path_params[7, :num_max_paths]
    
    if params[c.PARAMSET_SCENARIO_PARAMS][c.LOAD_FILE_SP_DOPPLER]:
        if path_params.shape[0]>8:
            user_data[c.OUT_PATH_DOP_VEL] =  path_params[8, :num_max_paths]
            user_data[c.OUT_PATH_DOP_ACC] =  path_params[9, :num_max_paths]
        else:
            user_data[c.OUT_PATH_DOP_VEL] =  np.zeros(path_params[7, :num_max_paths].shape)
            user_data[c.OUT_PATH_DOP_ACC] =  np.zeros(path_params[7, :num_max_paths].shape)
        
    path_verifier.verify_path(user_data[c.OUT_PATH_TOA], user_data[c.OUT_PATH_RX_POW])
    
    return user_data


def load_scenario_params(scenario_params_path):
    data = scipy.io.loadmat(scenario_params_path)
    scenario_params = {
                        c.PARAMSET_SCENARIO_PARAMS_CF: data[c.LOAD_FILE_SP_CF].astype(float).item(),
                        c.PARAMSET_SCENARIO_PARAMS_TX_POW: data[c.LOAD_FILE_SP_TX_POW].astype(float).item(),
                        c.PARAMSET_SCENARIO_PARAMS_NUM_BS: data[c.LOAD_FILE_SP_NUM_BS].astype(int).item(),
                        c.PARAMSET_SCENARIO_PARAMS_USER_GRIDS: data[c.LOAD_FILE_SP_USER_GRIDS].astype(int),
                        c.PARAMSET_SCENARIO_PARAMS_DOPPLER_EN: data[c.LOAD_FILE_SP_DOPPLER].astype(int).item(),
                        c.PARAMSET_SCENARIO_PARAMS_POLAR_EN: data[c.LOAD_FILE_SP_POLAR].astype(int).item()
                      }
    return scenario_params

# Loads the user and basestation dataset files
def load_ray_data(generation_idx, bs_id, params, user=True):
    
    # File Types and Directories
    ray_data = []
    rx_locs = []
    file_data = None
    if user:
        files = glob.glob(os.path.join(params[c.PARAMSET_SCENARIO_FIL], 'BS%i_UE*.mat'%bs_id))
        for file in files:
            filename = os.path.splitext(file)[0]
            user_ids = filename.split('_')[-1]
            file_start = int(user_ids.split('-')[0])
            file_end = int(user_ids.split('-')[1])
            users_in_file = np.logical_and(generation_idx >= file_start, generation_idx<file_end)
            if np.sum(users_in_file)>0:
                file_data = scipy.io.loadmat(file)
                for user in generation_idx[users_in_file]: # May remove this for loop with array indexing
                    if params['scenario_params']['dual_polar_available']:
                        if params['enable_dual_polar']:
                            ray_data.append(file_data['channels_VV'][0][user-file_start][0][0])
                            ray_data.append(file_data['channels_VH'][0][user-file_start][0][0])
                            ray_data.append(file_data['channels_HH'][0][user-file_start][0][0])
                            ray_data.append(file_data['channels_HV'][0][user-file_start][0][0])
                        else:
                            ray_data.append(file_data['channels_VV'][0][user-file_start][0][0])
                            
                    else:
                        ray_data.append(file_data['channels'][0][user-file_start][0][0])
                    rx_locs.append(file_data['rx_locs'][user-file_start])
    else:
        file = os.path.join(params[c.PARAMSET_SCENARIO_FIL], 'BS%i_BS.mat'%bs_id)
        file_data = scipy.io.loadmat(file)
        for bs_idx in generation_idx:
            if params['scenario_params']['dual_polar_available']:
                if params['enable_dual_polar']:
                    ray_data.append(file_data['channels_VV'][0][bs_idx][0][0])
                    ray_data.append(file_data['channels_VH'][0][bs_idx][0][0])
                    ray_data.append(file_data['channels_HH'][0][bs_idx][0][0])
                    ray_data.append(file_data['channels_HV'][0][bs_idx][0][0])
                else:
                    ray_data.append(file_data['channels_VV'][0][bs_idx][0][0])
                    
            else:
                ray_data.append(file_data['channels'][0][bs_idx][0][0])
            rx_locs.append(file_data['rx_locs'][bs_idx])
        
    ray_data = np.array(ray_data)
    if params['enable_dual_polar'] and params['scenario_params']['dual_polar_available']:
        ray_data = ray_data.reshape((4, -1))
        
    rx_locs = np.array(rx_locs)
    
    if file_data is not None and 'tx_loc' in file_data:
        tx_loc = file_data['tx_loc'].squeeze()
    # If tx location is not available in the data
    # pull it from the BS-BS file
    else:
        file = os.path.join(params[c.PARAMSET_SCENARIO_FIL], 'BS%i_BS.mat'%bs_id)
        file_data = scipy.io.loadmat(file)
        tx_loc = file_data['rx_locs'][bs_id-1][:3]
    return ray_data, rx_locs, tx_loc
