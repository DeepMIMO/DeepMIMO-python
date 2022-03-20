# -*- coding: utf-8 -*-
"""
% --- DeepMIMO Python: A Generic Dataset for mmWave and massive MIMO ----%
% Authors: Umut Demirhan
% DeepMIMO test script
% Date: 3/19/2022
"""

# # Import DeepMIMO and other needed libraries for this example
import DeepMIMO
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt

#%% Load and print the default parameters

parameters = DeepMIMO.default_params()
pprint(parameters, sort_dicts=False)


#%% Change parameters for the setup

# Scenario O1_60 extracted at the dataset_folder
parameters['scenario'] = 'O1_60'
parameters['dataset_folder'] = r'C:\Users\Umt\Desktop\scenarios'

parameters['num_paths'] = 10

# User rows 1-100
parameters['user_row_first'] = 1
parameters['user_row_last'] = 100

# Activate only the first basestation
parameters['active_BS'] = np.array([1]) 

parameters['OFDM']['bandwidth'] = 0.1 # 50 MHz
parameters['OFDM']['subcarriers'] = 512 # OFDM with 512 subcarriers
parameters['OFDM']['subcarriers_limit'] = 64 # Keep only first 64 subcarriers

parameters['ue_antenna']['shape'] = np.array([1, 1, 1]) # Single antenna
parameters['bs_antenna']['shape'] = np.array([1, 32, 1]) # ULA of 32 elements
#parameters['bs_antenna']['rotation'] = np.array([0, 30, 90]) # ULA of 32 elements
parameters['ue_antenna']['rotation'] = np.array([[0, 30], [30, 60], [60, 90]]) # ULA of 32 elements
#parameters['ue_antenna']['radiation_pattern'] = 'isotropic' 
#parameters['bs_antenna']['radiation_pattern'] = 'halfwave-dipole' 


#%% Generate and inspect the dataset
dataset = DeepMIMO.generate_data(parameters)

# Number of basestations
len(dataset)

# Keys of a basestation dictionary
dataset[0].keys()

# Keys of a channel
dataset[0]['user'].keys()

# Number of UEs
len(dataset[0]['user']['channel'])

# Shape of the channel matrix
dataset[0]['user']['channel'].shape

# Shape of BS 0 - UE 0 channel
dataset[0]['user']['channel'][0].shape

# Path properties of BS 0 - UE 0
pprint(dataset[0]['user']['paths'][0])


#%% Visualization of a channel matrix
plt.figure()
# Visualize channel magnitude response
# First, select indices of a user and bs
ue_idx = 0
bs_idx = 0
# Import channel
channel = dataset[bs_idx]['user']['channel'][ue_idx]
# Take only the first antenna pair
plt.imshow(np.abs(np.squeeze(channel).T))
plt.title('Channel Magnitude Response')
plt.xlabel('TX Antennas')
plt.ylabel('Subcarriers')


#%% Visualization of the UE positions and path-losses
loc_x = dataset[bs_idx]['user']['location'][:, 0]
loc_y = dataset[bs_idx]['user']['location'][:, 1]
loc_z = dataset[bs_idx]['user']['location'][:, 2]
pathloss = dataset[bs_idx]['user']['pathloss']
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
im = ax.scatter(loc_x, loc_y, loc_z, c=pathloss)
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('z (m)')

bs_loc_x = dataset[bs_idx]['basestation']['location'][:, 0]
bs_loc_y = dataset[bs_idx]['basestation']['location'][:, 1]
bs_loc_z = dataset[bs_idx]['basestation']['location'][:, 2]
ax.scatter(bs_loc_x, bs_loc_y, bs_loc_z, c='r')
ttl = plt.title('UE and BS Positions')

#%%
fig = plt.figure()
ax = fig.add_subplot()
im = ax.scatter(loc_x, loc_y, c=pathloss)
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
fig.colorbar(im, ax=ax)
ttl = plt.title('UE Grid Path-loss (dBm)')

