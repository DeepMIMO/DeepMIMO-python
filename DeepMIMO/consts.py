# -*- coding: utf-8 -*-
"""
DeepMIMOv2 Python Implementation

Description: Constant file

Authors: Umut Demirhan, Ahmed Alkhateeb
Date: 12/10/2021
"""

DICT_UE_IDX = 'user'
DICT_BS_IDX = 'basestation'

# NAME OF PARAMETER VARIABLES
PARAMSET_DATASET_FOLDER = 'dataset_folder'
PARAMSET_SCENARIO = 'scenario'
PARAMSET_DYNAMIC = 'dynamic_settings'
PARAMSET_DYNAMIC_FIRST = 'first_scene'
PARAMSET_DYNAMIC_LAST = 'last_scene'

PARAMSET_NUM_PATHS = 'num_paths'
PARAMSET_ACTIVE_BS = 'active_BS'
PARAMSET_USER_ROW_FIRST = 'user_row_first'
PARAMSET_USER_ROW_LAST = 'user_row_last'
PARAMSET_USER_ROW_SUBSAMP = 'row_subsampling'
PARAMSET_USER_SUBSAMP = 'user_subsampling'

PARAMSET_BS2BS = 'enable_BS2BS'
PARAMSET_FDTD = 'OFDM_channels' # TD/OFDM

PARAMSET_OFDM = 'OFDM'
PARAMSET_OFDM_SC_NUM = 'subcarriers'
PARAMSET_OFDM_SC_LIM = 'subcarriers_limit'
PARAMSET_OFDM_SC_SAMP = 'subcarriers_sampling'
PARAMSET_OFDM_BW = 'bandwidth'
PARAMSET_OFDM_BW_MULT = 1e9 # Bandwidth input is GHz, multiply by this
PARAMSET_OFDM_LPF = 'RX_filter'

PARAMSET_ANT_BS = 'bs_antenna'
PARAMSET_ANT_UE = 'ue_antenna'
PARAMSET_ANT_SHAPE = 'shape'
PARAMSET_ANT_SPACING = 'spacing'
PARAMSET_ANT_ROTATION = 'rotation'
PARAMSET_ANT_RAD_PAT = 'radiation_pattern'
PARAMSET_ANT_RAD_PAT_VALS = ['isotropic', 'halfwave-dipole'] 

# INNER VARIABLES
PARAMSET_ACTIVE_UE = 'active_UE'
PARAMSET_SCENARIO_FIL = 'scenario_files'
#PARAMSET_ACTIVE_USERS = 'active_user_idx'
PARAMSET_ANT_BS_DIFF = 'BS2BS_isnumpy' # Based on this paramater, the BS-BS channels won't be converted from a list of matrices to a single matrix

# SCENARIO PARAMS
PARAMSET_SCENARIO_PARAMS = 'scenario_params'
PARAMSET_SCENARIO_PARAMS_CF = 'carrier_freq'
PARAMSET_SCENARIO_PARAMS_TX_POW = 'tx_power'
PARAMSET_SCENARIO_PARAMS_NUM_BS = 'num_BS'
PARAMSET_SCENARIO_PARAMS_USER_GRIDS = 'user_grids'

# OUTPUT VARIABLES
OUT_CHANNEL = 'channel'
OUT_PATH = 'paths'
OUT_LOS = 'LoS'
OUT_LOC = 'location'
OUT_DIST = 'distance'
OUT_PL = 'pathloss'

OUT_PATH_NUM = 'num_paths'
OUT_PATH_DOD_PHI = 'DoD_phi'
OUT_PATH_DOD_THETA = 'DoD_theta'
OUT_PATH_DOA_PHI = 'DoA_phi'
OUT_PATH_DOA_THETA = 'DoA_theta'
OUT_PATH_PHASE = 'phase'
OUT_PATH_TOA = 'ToA'
OUT_PATH_RX_POW = 'power'
OUT_PATH_ACTIVE = 'active_paths'

# FILE LISTS - raytracing.load_ray_data()
LOAD_FILE_EXT = ['DoD', 'DoA', 'CIR', 'LoS', 'PL', 'Loc']
LOAD_FILE_EXT_FLATTEN =[1, 1, 1, 1, 0, 0]
LOAD_FILE_EXT_UE = ['DoD.mat', 'DoA.mat', 'CIR.mat', 'LoS.mat', 'PL.mat', 'Loc.mat']
LOAD_FILE_EXT_BS = ['DoD.BSBS.mat', 'DoA.BSBS.mat', 'CIR.BSBS.mat', 'LoS.BSBS.mat', 'PL.BSBS.mat', 'BSBS.RX_Loc.mat']

# TX LOCATION FILE VARIABLE NAME - load_scenario_params()
LOAD_FILE_TX_LOC = 'TX_Loc_array_full'

# SCENARIO PARAMS FILE VARIABLE NAMES - load_scenario_params()
LOAD_FILE_SP_EXT = '.params.mat'
LOAD_FILE_SP_CF = 'carrier_freq'
LOAD_FILE_SP_TX_POW = 'transmit_power'
LOAD_FILE_SP_NUM_BS = 'num_BS'
LOAD_FILE_SP_USER_GRIDS = 'user_grids'
