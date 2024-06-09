# -*- coding: utf-8 -*-
"""
DeepMIMOv3 Python Implementation

Description: Main generator script

Authors: Umut Demirhan, Ahmed Alkhateeb
Date: 12/10/2021
"""

import os
import copy
import numpy as np

import DeepMIMOv3.consts as c
from DeepMIMOv3.construct_deepmimo import generate_MIMO_channel, generate_MIMO_channel_rx_ind
from DeepMIMOv3.utils import safe_print
from DeepMIMOv3.params import default_params

def generate_data(ext_params):
    
    np.random.seed(1001)
    
    params = validate_params(copy.deepcopy(ext_params))
    
    # If dynamic scenario
    if is_dynamic_scenario(params):
        scene_list = params[c.PARAMSET_DYNAMIC_SCENES]
        num_of_scenes = len(scene_list)
        dataset = []
        for scene_i in range(num_of_scenes):
            scene = scene_list[scene_i]
            params[c.PARAMSET_SCENARIO_FIL] = os.path.join(
                                        os.path.abspath(params[c.PARAMSET_DATASET_FOLDER]), 
                                        params[c.PARAMSET_SCENARIO],
                                        'scene_' + str(scene)
                                        )
            print('\nScene %i/%i' % (scene_i+1, num_of_scenes))
            dataset.append(generate_scene_data(params))
    # If static scenario
    else:
        params[c.PARAMSET_SCENARIO_FIL] = os.path.join(
                                    os.path.abspath(params[c.PARAMSET_DATASET_FOLDER]), 
                                    params[c.PARAMSET_SCENARIO]
                                    )
        dataset = generate_scene_data(params)
    return dataset
        
def generate_scene_data(params):
    num_active_bs = len(params[c.PARAMSET_ACTIVE_BS])
    dataset = [{c.DICT_UE_IDX: dict(), c.DICT_BS_IDX: dict(), c.OUT_LOC: None} for x in range(num_active_bs)]
    
    for i in range(num_active_bs):
        bs_indx = params[c.PARAMSET_ACTIVE_BS][i]
        
        safe_print('\nBasestation %i' % bs_indx)
        
        safe_print('\nUE-BS Channels')
        dataset[i][c.DICT_UE_IDX], dataset[i][c.OUT_LOC] = params['raytracing_fn'](bs_indx, params, user=True)
        
        if params['scenario_params']['dual_polar_available'] and params['enable_dual_polar']:
            for polar_str in ['VV', 'VH', 'HH', 'HV']:
                dataset[i][c.DICT_UE_IDX][polar_str][c.OUT_CHANNEL], dataset[i][c.DICT_UE_IDX][polar_str][c.OUT_LOS] = generate_MIMO_channel(dataset[i][c.DICT_UE_IDX][polar_str][c.OUT_PATH], 
                                                                                                                                             params, 
                                                                                                                                             params[c.PARAMSET_ANT_BS][i], 
                                                                                                                                             params[c.PARAMSET_ANT_UE]
                                                                                                                                             )
        else:
            dataset[i][c.DICT_UE_IDX][c.OUT_CHANNEL], dataset[i][c.DICT_UE_IDX][c.OUT_LOS] = generate_MIMO_channel(dataset[i][c.DICT_UE_IDX][c.OUT_PATH], 
                                                                                                                   params, 
                                                                                                                   params[c.PARAMSET_ANT_BS][i], 
                                                                                                                   params[c.PARAMSET_ANT_UE]
                                                                                                                   )
                
        if params[c.PARAMSET_BS2BS]:
            safe_print('\nBS-BS Channels')
            
            dataset[i][c.DICT_BS_IDX], _ = params['raytracing_fn'](bs_indx, params, user=False)
            
            if params['scenario_params']['dual_polar_available'] and params['enable_dual_polar']:
                for polar_str in ['VV', 'VH', 'HH', 'HV']:
                    dataset[i][c.DICT_BS_IDX][polar_str][c.OUT_CHANNEL], dataset[i][c.DICT_BS_IDX][polar_str][c.OUT_LOS] = generate_MIMO_channel_rx_ind(dataset[i][c.DICT_BS_IDX][polar_str][c.OUT_PATH], 
                                                                                                                                                        params, 
                                                                                                                                                        params[c.PARAMSET_ANT_BS][i], 
                                                                                                                                                        params[c.PARAMSET_ANT_BS]
                                                                                                                                                        )
                if not params[c.PARAMSET_ANT_BS_DIFF]:
                    dataset[i][c.DICT_BS_IDX][polar_str][c.OUT_CHANNEL], dataset[i][c.DICT_BS_IDX][polar_str][c.OUT_LOS] = np.stack(dataset[i][c.DICT_BS_IDX][polar_str][c.OUT_CHANNEL], axis=0)
            else:
                dataset[i][c.DICT_BS_IDX][c.OUT_CHANNEL], dataset[i][c.DICT_BS_IDX][c.OUT_LOS] = generate_MIMO_channel_rx_ind(dataset[i][c.DICT_BS_IDX][c.OUT_PATH], 
                                                                                                                              params, 
                                                                                                                              params[c.PARAMSET_ANT_BS][i], 
                                                                                                                              params[c.PARAMSET_ANT_BS])
            
                if not params[c.PARAMSET_ANT_BS_DIFF]:
                    dataset[i][c.DICT_BS_IDX][c.OUT_CHANNEL], dataset[i][c.DICT_BS_IDX][c.OUT_LOS] = np.stack(dataset[i][c.DICT_BS_IDX][c.OUT_CHANNEL], axis=0)
    return dataset

# TODO: Move validation into another script
def validate_params(params):

    additional_keys = compare_two_dicts(params, default_params())
    if len(additional_keys):
        print('The following parameters seem unnecessary:')
        print(additional_keys)
    
    params['dynamic_scenario'] = is_dynamic_scenario(params)
    if params['dynamic_scenario']:
        params['user_rows'] = np.array([0])
    params['data_version'] = check_data_version(params)
    params[c.PARAMSET_SCENARIO_PARAMS_PATH] = get_scenario_params_path(params)
    if params['data_version'] == 'v2':
        from DeepMIMOv3.raytracing_v2 import load_scenario_params, read_raytracing
    elif params['data_version'] == 'v3':
        from DeepMIMOv3.raytracing_v3 import load_scenario_params, read_raytracing
    params['raytracing_fn'] = read_raytracing
    params[c.PARAMSET_SCENARIO_PARAMS] = load_scenario_params(params[c.PARAMSET_SCENARIO_PARAMS_PATH])
    
    # Active user IDs and related parameter
    assert params[c.PARAMSET_USER_SUBSAMP] > 0 and params[c.PARAMSET_USER_SUBSAMP] <= 1, 'The subsampling parameter \'%s\' needs to be in (0, 1]'%c.PARAMSET_USER_SUBSAMP
    params[c.PARAMSET_ACTIVE_UE] = find_users_from_rows(params)
    
    # BS antenna format
    params[c.PARAMSET_ANT_BS_DIFF] = True
    if type(params[c.PARAMSET_ANT_BS]) is dict: # Replicate BS Antenna for each active BS in a list
        ant = params[c.PARAMSET_ANT_BS]
        params[c.PARAMSET_ANT_BS] = []
        for i in range(len(params[c.PARAMSET_ACTIVE_BS])):
            params[c.PARAMSET_ANT_BS].append(ant)
    else:
        if len(params[c.PARAMSET_ACTIVE_BS]) == 1:
            params[c.PARAMSET_ANT_BS_DIFF] = False 
            
    # BS Antenna Rotation
    for i in range(len(params[c.PARAMSET_ACTIVE_BS])):
        if c.PARAMSET_ANT_ROTATION in params[c.PARAMSET_ANT_BS][i].keys() and params[c.PARAMSET_ANT_BS][i][c.PARAMSET_ANT_ROTATION] is not None:
            rotation_shape = params[c.PARAMSET_ANT_BS][i][c.PARAMSET_ANT_ROTATION].shape
            assert  (len(rotation_shape) == 1 and rotation_shape[0] == 3) \
                    ,'The BS antenna rotation must be a 3D vector'
                    
        else:
            params[c.PARAMSET_ANT_BS][i][c.PARAMSET_ANT_ROTATION] = None                                            
      
    # UE Antenna Rotation
    if c.PARAMSET_ANT_ROTATION in params[c.PARAMSET_ANT_UE].keys() and params[c.PARAMSET_ANT_UE][c.PARAMSET_ANT_ROTATION] is not None:
        rotation_shape = params[c.PARAMSET_ANT_UE][c.PARAMSET_ANT_ROTATION].shape
        assert (len(rotation_shape) == 1 and rotation_shape[0] == 3) or \
                (len(rotation_shape) == 2 and rotation_shape[0] == 3 and rotation_shape[1] == 2) or \
                (rotation_shape[0] == len(params[c.PARAMSET_ACTIVE_UE])) \
                ,'The UE antenna rotation must either be a 3D vector for constant values or 3 x 2 matrix for random values'
                
        if len(rotation_shape) == 1 and rotation_shape[0] == 3:
            rotation = np.zeros((len(params[c.PARAMSET_ACTIVE_UE]), 3))
            rotation[:] =  params[c.PARAMSET_ANT_UE][c.PARAMSET_ANT_ROTATION]
            params[c.PARAMSET_ANT_UE][c.PARAMSET_ANT_ROTATION] = rotation
        elif (len(rotation_shape) == 2 and rotation_shape[0] == 3 and rotation_shape[1] == 2):
            params[c.PARAMSET_ANT_UE][c.PARAMSET_ANT_ROTATION] = np.random.uniform(
                              params[c.PARAMSET_ANT_UE][c.PARAMSET_ANT_ROTATION][:, 0], 
                              params[c.PARAMSET_ANT_UE][c.PARAMSET_ANT_ROTATION][:, 1], 
                              (len(params[c.PARAMSET_ACTIVE_UE]), 3))
    else:
        params[c.PARAMSET_ANT_UE][c.PARAMSET_ANT_ROTATION] = np.array([None] * len(params[c.PARAMSET_ACTIVE_UE])) # List of None
     
    # BS Antenna Radiation Pattern
    for i in range(len(params[c.PARAMSET_ACTIVE_BS])):
        if c.PARAMSET_ANT_RAD_PAT in params[c.PARAMSET_ANT_BS][i].keys():
            assert params[c.PARAMSET_ANT_BS][i][c.PARAMSET_ANT_RAD_PAT] in c.PARAMSET_ANT_RAD_PAT_VALS, 'The antenna radiation pattern for BS-%i must have one of the following values: [%s]' %(i, c.PARAMSET_ANT_RAD_PAT_VALS.join(', '))
        else:
            params[c.PARAMSET_ANT_BS][i][c.PARAMSET_ANT_RAD_PAT] = c.PARAMSET_ANT_RAD_PAT_VALS[0]
                     
    # UE Antenna Radiation Pattern
    if c.PARAMSET_ANT_RAD_PAT in params[c.PARAMSET_ANT_UE].keys():
        assert params[c.PARAMSET_ANT_UE][c.PARAMSET_ANT_RAD_PAT] in c.PARAMSET_ANT_RAD_PAT_VALS, 'The antenna radiation pattern for UEs must have one of the following values: [%s]' %(c.PARAMSET_ANT_RAD_PAT_VALS.join(', '))
    else:
        params[c.PARAMSET_ANT_UE][c.PARAMSET_ANT_RAD_PAT] = c.PARAMSET_ANT_RAD_PAT_VALS[0]
                                             
    return params


# Generate the set of users to be activated
def find_users_from_rows(params):

    def rand_perm_per(vector, percentage):
        if percentage == 1: return vector
        num_of_subsampled = round(len(vector)*percentage)
        if num_of_subsampled < 1: num_of_subsampled = 1 
        subsampled = np.arange(len(vector))
        np.random.shuffle(subsampled)
        subsampled = vector[subsampled[:num_of_subsampled]]
        subsampled = np.sort(subsampled)
        return subsampled
    
    def get_user_ids(row, grids):
        row = row + 1
        row_prev_ids = np.sum((row > grids[:, 1])*(grids[:, 1] - grids[:, 0] + 1)*grids[:, 2])
        row_cur_ind = (grids[:, 1] >= row) * (row >= grids[:, 0])
        row_cur_start = row - grids[row_cur_ind, 0][0]
        users_in_row = grids[:, 2][row_cur_ind][0]

        # column-oriented grid
        if grids.shape[1] == 4 and grids[row_cur_ind, 3][0]: 
            users_in_col = (grids[:, 1]-grids[:, 0]+1)[row_cur_ind][0]
            user_ids = row_prev_ids + row_cur_start + np.arange(0, users_in_col*users_in_row, users_in_col)
        # row-oriented grid
        else: 
            row_curr_ids = row_cur_start * users_in_row
            user_ids = row_prev_ids + row_curr_ids + np.arange(users_in_row)
            
        return user_ids
    
    grids = params[c.PARAMSET_SCENARIO_PARAMS][c.PARAMSET_SCENARIO_PARAMS_USER_GRIDS]
    rows = params[c.PARAMSET_USER_ROWS]
    
    user_ids = np.array([], dtype=int)
    for row in rows:
        user_ids_row = get_user_ids(row, grids)
        user_ids_row = rand_perm_per(user_ids_row, params[c.PARAMSET_USER_SUBSAMP])
        user_ids = np.concatenate((user_ids, user_ids_row))
    
    return user_ids

def is_dynamic_scenario(params):
    dynamic = 'dyn' in params[c.PARAMSET_SCENARIO]
    return dynamic

def check_data_version(params):
    v3_params_path = os.path.join(os.path.abspath(params[c.PARAMSET_DATASET_FOLDER]), 
                                    params[c.PARAMSET_SCENARIO],
                                    'params.mat')
    if os.path.isfile(v3_params_path):
        return 'v3'
    else:
        return 'v2'
    
def get_scenario_params_path(params):
    if params['data_version'] == 'v2':
        if params['dynamic_scenario']:
            params_path = os.path.join(
                                        os.path.abspath(params[c.PARAMSET_DATASET_FOLDER]), 
                                        params[c.PARAMSET_SCENARIO],
                                        'scene_' + str(params[c.PARAMSET_DYNAMIC_SCENES][0]), # 'scene_i' folder
                                        params[c.PARAMSET_SCENARIO] + c.LOAD_FILE_SP_EXT
                                        )
        else:
            params_path = os.path.join(
                                        os.path.abspath(params[c.PARAMSET_DATASET_FOLDER]), 
                                        params[c.PARAMSET_SCENARIO], 
                                        params[c.PARAMSET_SCENARIO] + c.LOAD_FILE_SP_EXT
                                      )
    elif params['data_version'] == 'v3':
        params_path = os.path.join(
                                   os.path.abspath(params[c.PARAMSET_DATASET_FOLDER]), 
                                   params[c.PARAMSET_SCENARIO],
                                   'params.mat'
                                  )
    else:
        raise NotImplementedError
        
    return params_path

def compare_two_dicts(dict1, dict2):
    
    additional_keys = dict1.keys() - dict2.keys()
    for key, item in dict1.items():
        if isinstance(item, dict):
            if key in dict2:
                additional_keys = additional_keys | compare_two_dicts(dict1[key], dict2[key])

    return additional_keys

