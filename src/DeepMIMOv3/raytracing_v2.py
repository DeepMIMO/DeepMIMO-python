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

def read_raytracing(bs_id, params, user=True):
    
    scenario_files = os.path.join(params[c.PARAMSET_SCENARIO_FIL], params[c.PARAMSET_SCENARIO])
    params[c.PARAMSET_SCENARIO_PARAMS] = load_scenario_params(params[c.PARAMSET_SCENARIO_PARAMS_PATH])
    
    if user:
        generation_idx = params[c.PARAMSET_ACTIVE_UE] # Active User IDX
    else:
        generation_idx = params[c.PARAMSET_ACTIVE_BS]-1 # Active BS IDX
        
    ray_data = load_ray_data(scenario_files, bs_id, user=user)
    data = extract_data_from_ray(ray_data, generation_idx, params)
    
    bs_loc = load_bs_loc(scenario_files, bs_id)
    return data, bs_loc

# Extracts the information to a dictionary from the loaded data file
# for the users given in IDs, and with maximum number of paths
def extract_data_from_ray(ray_data, ids, params):
    
    path_verifier = PathVerifier(params)
    
    num_channels = len(ids)
    
    pointer = 1 # First user ID
    
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
    data = {c.OUT_PATH: np.array([dict(path_dict) for j in range(len(ids))]),
            c.OUT_LOS : np.zeros(num_channels, dtype=int),
            c.OUT_LOC : np.zeros((num_channels, 3)),
            c.OUT_DIST : np.zeros(num_channels),
            c.OUT_PL : np.zeros(num_channels)
            }
    
    j = 0
    for user in tqdm(range(max(ids)+1), desc='Reading ray-tracing'):
        pointer += 1 # Number of Paths
        num_paths_available = int(ray_data[c.LOAD_FILE_EXT[0]][pointer]) # DoD file
        pointer += 1 # First Path
        if user in ids:
            num_paths_read = min(num_paths_available, params[c.PARAMSET_NUM_PATHS])
            path_limited_data_length = num_paths_read*4;
                
            if num_paths_available>0:
                data[c.OUT_PATH][j] = load_variables(num_paths_read=num_paths_read, 
                                                     path_DoD=ray_data[c.LOAD_FILE_EXT[0]][(pointer):(pointer+path_limited_data_length)], 
                                                     path_DoA=ray_data[c.LOAD_FILE_EXT[1]][(pointer):(pointer+path_limited_data_length)], 
                                                     path_CIR=ray_data[c.LOAD_FILE_EXT[2]][(pointer):(pointer+path_limited_data_length)], 
                                                     LoS_status=ray_data[c.LOAD_FILE_EXT[3]][user+1],
                                                     path_verifier=path_verifier, 
                                                     params=params)
                data[c.OUT_PL][j] = ray_data[c.LOAD_FILE_EXT[4]][user, 1]
                
            data[c.OUT_LOS][j] = ray_data[c.LOAD_FILE_EXT[3]][user+1]
            data[c.OUT_LOC][j] = ray_data[c.LOAD_FILE_EXT[5]][user, 1:4]
            data[c.OUT_DIST][j] = ray_data[c.LOAD_FILE_EXT[4]][user, 0]
            j += 1
            
        pointer += num_paths_available*4
        
    path_verifier.notify()
    
    # The reading operation of the raytracing is linear
    # Therefore, it is re-ordered to return in the same order of user IDs
    rev_argsort = np.empty(ids.shape, dtype=np.intp)
    rev_argsort[np.argsort(ids)] = np.arange(len(ids))
    for dict_key in data.keys():
        data[dict_key] = data[dict_key][rev_argsort]
    return data

# Split variables into a dictionary
def load_variables(num_paths_read, path_DoD, path_DoA, path_CIR, LoS_status, path_verifier, params):
    user_data = dict()
    user_data[c.OUT_PATH_NUM] = num_paths_read
    user_data[c.OUT_PATH_DOD_PHI] = path_DoD[1::4]
    user_data[c.OUT_PATH_DOD_THETA] = path_DoD[2::4]
    user_data[c.OUT_PATH_DOA_PHI] = path_DoA[1::4]
    user_data[c.OUT_PATH_DOA_THETA] = path_DoA[2::4]
    user_data[c.OUT_PATH_PHASE] = path_CIR[1::4]
    user_data[c.OUT_PATH_TOA] = path_CIR[2::4]
    
    aux = np.zeros_like(user_data[c.OUT_PATH_TOA])
    aux[0] = LoS_status
    user_data[c.OUT_PATH_LOS] = aux
    
    user_data[c.OUT_PATH_RX_POW] = dbm2pow(path_CIR[3::4] + 30 - params[c.PARAMSET_SCENARIO_PARAMS][c.PARAMSET_SCENARIO_PARAMS_TX_POW])
    path_verifier.verify_path(user_data[c.OUT_PATH_TOA], user_data[c.OUT_PATH_RX_POW])
    return user_data


def load_scenario_params(scenario_params_path):
    data = scipy.io.loadmat(scenario_params_path)
    scenario_params = {
                        c.PARAMSET_SCENARIO_PARAMS_CF: data[c.LOAD_FILE_SP_CF].astype(float).item(),
                        c.PARAMSET_SCENARIO_PARAMS_TX_POW: data[c.LOAD_FILE_SP_TX_POW].astype(float).item(),
                        c.PARAMSET_SCENARIO_PARAMS_NUM_BS: data[c.LOAD_FILE_SP_NUM_BS].astype(int).item(),
                        c.PARAMSET_SCENARIO_PARAMS_USER_GRIDS: data[c.LOAD_FILE_SP_USER_GRIDS].astype(int),
                        c.PARAMSET_SCENARIO_PARAMS_DOPPLER_EN: 0,
                        c.PARAMSET_SCENARIO_PARAMS_POLAR_EN: 0
                      }
    return scenario_params

def load_bs_loc(scenario_files, bs_id):
    TX_loc_file = scenario_files + '.TX_Loc.mat'
    data = scipy.io.loadmat(TX_loc_file)
    return data[list(data.keys())[3]].astype(float)[bs_id-1, 1:4]

# Loads the user and basestation dataset files
def load_ray_data(scenario_files, bs_id, user=True):
    # File Types and Directories
    file_list = c.LOAD_FILE_EXT
    file_list_reshape = c.LOAD_FILE_EXT_FLATTEN
    
    if user:
        file_loc = [scenario_files +  '.%i.' % bs_id + c.LOAD_FILE_EXT_UE[0], 
                    scenario_files +  '.%i.' % bs_id + c.LOAD_FILE_EXT_UE[1], 
                    scenario_files +  '.%i.' % bs_id + c.LOAD_FILE_EXT_UE[2], 
                    scenario_files +  '.%i.' % bs_id + c.LOAD_FILE_EXT_UE[3],
                    scenario_files +  '.%i.' % bs_id + c.LOAD_FILE_EXT_UE[4],
                    scenario_files +  '.' + c.LOAD_FILE_EXT_UE[5]]
    else: # Basestation
        file_loc = [scenario_files +  '.%i.' % bs_id + c.LOAD_FILE_EXT_BS[0], 
                    scenario_files +  '.%i.' % bs_id + c.LOAD_FILE_EXT_BS[1], 
                    scenario_files +  '.%i.' % bs_id + c.LOAD_FILE_EXT_BS[2], 
                    scenario_files +  '.%i.' % bs_id + c.LOAD_FILE_EXT_BS[3],
                    scenario_files +  '.%i.' % bs_id + c.LOAD_FILE_EXT_BS[4], 
                    scenario_files +  '.' + c.LOAD_FILE_EXT_BS[5]]
    
    # Load files
    ray_data = dict.fromkeys(file_list)
    for i in range(len(file_list)):
        data = scipy.io.loadmat(file_loc[i])
        ray_data[file_list[i]] = data[list(data.keys())[3]]
        if file_list_reshape[i]:
            ray_data[file_list[i]] = ray_data[file_list[i]].reshape(-1) # 3rd key is the data

    return ray_data