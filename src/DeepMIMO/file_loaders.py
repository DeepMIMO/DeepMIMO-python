# -*- coding: utf-8 -*-
"""
DeepMIMOv2 Python Implementation

Description: File Loader Functions

Authors: Umut Demirhan, Ahmed Alkhateeb
Date: 7/2/2022
"""


import scipy.io
import DeepMIMO.consts as c

def load_scenario_params(scenario_files):
    file_loc = scenario_files + c.LOAD_FILE_SP_EXT # Scenario parameters file
    data = scipy.io.loadmat(file_loc)
    scenario_params = {c.PARAMSET_SCENARIO_PARAMS_CF: data[c.LOAD_FILE_SP_CF].astype(float).item(),
                       c.PARAMSET_SCENARIO_PARAMS_TX_POW: data[c.LOAD_FILE_SP_TX_POW].astype(float).item(),
                       c.PARAMSET_SCENARIO_PARAMS_NUM_BS: data[c.LOAD_FILE_SP_NUM_BS].astype(int).item(),
                       c.PARAMSET_SCENARIO_PARAMS_USER_GRIDS: data[c.LOAD_FILE_SP_USER_GRIDS].astype(int)
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