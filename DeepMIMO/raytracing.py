# -*- coding: utf-8 -*-
"""
DeepMIMOv2 Python Implementation

Description: Read scenario files

Authors: Umut Demirhan, Ahmed Alkhateeb
Date: 12/10/2021
"""


import numpy as np
from tqdm import tqdm
import DeepMIMO.consts as c
from DeepMIMO.utils import safe_print
from DeepMIMO.file_loaders import load_scenario_params, load_bs_loc, load_ray_data

def read_raytracing(bs_id, params, user=True):
    
    scenario_files = params[c.PARAMSET_SCENARIO_FIL]
    params[c.PARAMSET_SCENARIO_PARAMS] = load_scenario_params(scenario_files)
    
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
                 c.OUT_PATH_ACTIVE: []
                 }
    data = {c.OUT_PATH: [dict(path_dict) for j in range(len(ids))],
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
                                                     path_verifier=path_verifier, 
                                                     params=params)
                data[c.OUT_PL][j] = ray_data[c.LOAD_FILE_EXT[4]][user, 1]
                
            data[c.OUT_LOS][j] = ray_data[c.LOAD_FILE_EXT[3]][user+1]
            data[c.OUT_LOC][j] = ray_data[c.LOAD_FILE_EXT[5]][user, 1:4]
            data[c.OUT_DIST][j] = ray_data[c.LOAD_FILE_EXT[4]][user, 0]
            j += 1
            
        pointer += num_paths_available*4
        
    path_verifier.notify()
    return data

# Split variables into a dictionary
def load_variables(num_paths_read, path_DoD, path_DoA, path_CIR, path_verifier, params):
    user_data = dict()
    user_data[c.OUT_PATH_NUM] = num_paths_read
    user_data[c.OUT_PATH_DOD_PHI] = path_DoD[1::4]
    user_data[c.OUT_PATH_DOD_THETA] = path_DoD[2::4]
    user_data[c.OUT_PATH_DOA_PHI] = path_DoA[1::4]
    user_data[c.OUT_PATH_DOA_THETA] = path_DoA[2::4]
    user_data[c.OUT_PATH_PHASE] = path_CIR[1::4]
    user_data[c.OUT_PATH_TOA] = path_CIR[2::4]
    user_data[c.OUT_PATH_RX_POW] = dbm2pow(path_CIR[3::4] + 30 - params[c.PARAMSET_SCENARIO_PARAMS][c.PARAMSET_SCENARIO_PARAMS_TX_POW])
    path_verifier.verify_path(user_data[c.OUT_PATH_TOA], user_data[c.OUT_PATH_RX_POW])
    return user_data
    
# Determine active paths with the given configurations
# (For OFDM, only the paths within DS are activated)
class PathVerifier:
    def __init__(self, params):
        self.params = params
        Ts = 1 / (params[c.PARAMSET_OFDM][c.PARAMSET_OFDM_BW]*c.PARAMSET_OFDM_BW_MULT)
        self.FFT_duration = params[c.PARAMSET_OFDM][c.PARAMSET_OFDM_SC_NUM] * Ts
        self.max_ToA = 0
        self.path_ratio_FFT = []
    
    def verify_path(self, ToA, power):
        if self.params[c.PARAMSET_FDTD]: # OFDM CH
            m_toa = np.max(ToA)
            self.max_ToA = max(self.max_ToA, m_toa)
            
            if m_toa > self.FFT_duration:
                violating_paths = ToA > self.FFT_duration
                self.path_ratio_FFT.append( sum(power[violating_paths])/sum(power) )
                        
    def notify(self):
        avg_ratio_FFT = 0
        if len(self.path_ratio_FFT) != 0:
            avg_ratio_FFT = np.mean(self.path_ratio_FFT)*100
            
        if self.params[c.PARAMSET_FDTD]: # IF OFDM
            if self.max_ToA > self.FFT_duration and avg_ratio_FFT >= 1.:
                safe_print('ToA of some paths of %i channels with an average total power of %.2f%% exceed the useful OFDM symbol duration and are clipped.' % (len(self.path_ratio_FFT), avg_ratio_FFT))
            
def dbm2pow(val):
    return 10**(val/10 - 3)