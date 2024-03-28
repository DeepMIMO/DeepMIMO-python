# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 2023

@author: Umut Demirhan
"""

import DeepMIMOv3

import numpy as np

parameters = DeepMIMOv3.default_params()

# parameters['scenario'] = 'O1_60'
parameters['dataset_folder'] = r'C:\Users\Umt\Documents\GitHub\DeepMIMO-matlab\Raytracing_scenarios'

parameters['scenario'] = 'Boston5G_3p5_polar'
# parameters['dataset_folder'] = r'C:\Users\Umt\Desktop\Boston5G_3p5_RIS'

# parameters['scenario'] = 'Boston5G_3p5_v2' # Dual-polar
# parameters['dataset_folder'] = r'C:\Users\Umt\Desktop\Boston5G_3p5_small'

# parameters['scenario'] = 'dyn3'
# parameters['dataset_folder'] = r'C:\Users\Umt\Desktop\dynamic_scenario'

parameters['enable_BS2BS'] = 0
parameters['RX_filter'] = 0

parameters['dynamic_scenario_scenes'] = np.arange(50, 55)
parameters['user_rows'] = np.arange(3)#np.array([0])

parameters['enable_dual_polar'] = 1
parameters['enable_doppler'] = 0

# To activate the half of the users in each selected row randomly, set
parameters['user_subsampling'] = 1

parameters['active_BS'] = np.array([1])

rotation = -60
# Define 2 different antennas:
antenna1 = {
            'shape': np.array([32, 1]),
            'spacing': 0.5,
            'rotation': np.array([0, 0, rotation]),
            'FoV': np.array([180, 180])
            }
antenna2 = {
            'shape': np.array([1, 1]),
            'spacing': 0.5,
            'rotation': np.array([0, 0, 0]),
            'FoV': np.array([180, 180])
            }
parameters['bs_antenna'] = [antenna1, antenna2]

parameters['OFDM']['selected_subcarrier'] = np.array([0])
dataset = DeepMIMOv3.generate_data(parameters)
dataset[0]['basestation']['channel'][1]

dataset[0]['basestation']['channel'][1].squeeze() - dataset[1]['basestation']['channel'][0].squeeze()


import matplotlib.pyplot as plt
beam_angles = np.linspace(-np.pi/2, np.pi/2, 128)
W = np.exp(-1j*np.pi*np.arange(32).reshape((-1, 1))@np.sin(beam_angles).reshape((1, -1)))
channel = dataset[0]['basestation']['channel'][1].squeeze()
antenna_gains = np.abs(channel @ W)
plt.plot(np.rad2deg(beam_angles), antenna_gains, label='rotation %i'%rotation)
plt.legend()
plt.grid()
plt.ylabel('Beam Gain')
plt.xlabel('Beam Angle')