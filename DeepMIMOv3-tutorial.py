# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 00:26:51 2024

@author: Umt
"""
# %% [markdown]
# # In this tutorial, we aim to use Phoenix scenario from the DeepMIMO website
# 
# ## In the design, the aim is to create dataset for the position based channel prediction task:
# - **Task:** Given user position, predict channel between the user and basestation
#     - **Input:** User Position
#     - **Output:** User-BS Channel
# 
# ## In the following:
# - We adjust the parameters and generate a dataset
# - Select a basestation
# - Check the user channels' LOS status
# - Select a subset of users with at least a single path
# - Finally, collect the positions and channels of these users for the machine learning task

# %%
# Import packages

# DeepMIMOv3 (and install through pip if not available) 
import DeepMIMOv3 as DeepMIMOv3
    
# Numpy and Matplotlib
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint

# %%
# Load the default parameters
parameters = DeepMIMOv3.default_params()

# Print the default parameters
pprint(parameters)

# %% [markdown]
# A brief summary for the parameters ([For the detailed descriptions, please go to the version page]()):
# 
# | Parameter              | Default Value          | Description                                                                        |
# |------------------------|----------------------------|---------------------------------------------------|
# | **dataset_folder**     | ./Raytracing_scenarios     | Folder of the datasets                                                            |
# | **scenario**           | O1_60                      | Scenario name - The .mat files of the scenario need to be in the path 'dataset_folder'/'scenario_name' |
# |                        |                      |    
# | **active_BS**          | [1]                        | The ID of the basestations                                                        |
# | **user_rows**          | [1]                        | User rows to be generated                                                          |
# | **user_subsampling**   | 1                          | Random subsampling rate for users - 1 generates all the users                      |
# | **dynamic_scenario_scenes** | [1]                   | Scenes to be generated in a dynamic scenario                                       |
# |                        |                      |    
# | **enable_BS2BS**       | 1                          | Enable BS to BS channels                                                           |
# | **enable_doppler**     | 0                          | Enable Doppler shift (if available in the scenario)                                |
# | **enable_dual_polar**  | 0                          | Enable dual cross-polarized antennas (if available in the scenario)                |
# |                        |                      |    
# | **num_paths**          | 5                          | Number of maximum paths                                                            |
# | **OFDM_channels**      | 1                          | Generate OFDM (True) or time domain channels (False)                               |
# | **OFDM**               |                            | OFDM parameters (only applies if OFDM_channels is True)                            |
# |   - **subcarriers**    | 512                        | Total number of subcarriers                                                        |
# |   - **selected_subcarriers** | [0]                  | Subcarriers to be generated                                                        |
# |   - **bandwidth**      | 0.05                       | Bandwidth                                                                          |
# |   - **RX_filter**      | 0                          | Receive filter                                                                     |
# |                        |                      |    
# | **bs_antenna**/**ue_antenna**         |                            | BS/UE antenna properties                                                              |
# |   - **FoV**            | [360, 180]                 | Antenna FoV Horizontal in [0,360] and vertical in [0, 180]                         |
# |   - **radiation_pattern** | isotropic              | Radiation pattern applied to the antenna, in ['isotropic', 'halfwave-dipole']       |
# |   - **rotation**       | [0, 0, 0]                  | Rotation of the antenna - in compliance with 38.901                                |
# |   - **shape**          | [8, 4]                     | UPA panel shape in the shape of (horizontal elements, vertical elements)           |
# |   - **spacing**        | 0.5                        | Antenna spacing                                                                    |

# %% [markdown]
# Scenario folders are extracted into "./Raytracing_scenarios"
# 
# Specifically, the phoenix scenario zip file is extracted into this folder. The scenario *.mat* files are inside "./Raytracing_scenarios/city_4_phoenix".
# 
# We set these parameters accordingly:

# %%
parameters['scenario'] = 'city_4_phoenix'
# parameters['dynamic_scenario_scenes'] = np.arange(475, 476)

# %% [markdown]
# There are 3 basestation in the scenario. Let us generate the data for all 3 of them.

# %%
parameters['active_BS'] = np.array([1, 2])

# %% [markdown]
# We want to generate all the users to visualize and subselect. There are 79 rows of users with 86 users in each row, as described in [the scenario page](https://www.deepmimo.net/scenarios/deepmimo-city-scenario4/)

# %%
parameters['user_rows'] = np.arange(5)

# %% [markdown]
# In this scenario, we consider a MISO case, i.e., single UE antenna with a 8-element ULA at the basestation

# %%
parameters['ue_antenna']['shape'] = np.array([1, 1])
parameters['bs_antenna']['shape'] = np.array([8, 1])

# %% [markdown]
# We now can generate the dataset with the adjusted parameters:

# %%
parameters['OFDM_channels'] = 0
parameters['OFDM']['RX_filter'] = 1
parameters['enable_doppler'] = 0
# %%
pprint(parameters)
dataset = DeepMIMOv3.generate_data(parameters)

# %% [markdown]
# Next, we visualize the LoS status (if the channel has -1: no path, 0: pnly NLoS paths, 1: LoS path) of the user channels. For this purpose, we need the LoS status of the user channels and position. From the output section of [the DeepMIMOv3-python page](https://www.deepmimo.net/versions/deepmimo-v3-python/), we check the commands needed to access these parameters, and collect them in new variables for plotting.

# %%
def plot_LoS_status(bs_location, user_locations, user_LoS, bs_title_idx=-1):
    LoS_map = {-1: ('r', 'No Path'), 0: ('b', 'NLoS'), 1: ('g', 'LoS')}
    
    plt.figure()
    for unique_LoS_status in LoS_map.keys():
    # Plot different status one by one to assign legend labels
        users_w_key = user_LoS==unique_LoS_status
        plt.scatter(user_locations[users_w_key, 0], 
                    user_locations[users_w_key, 1], 
                    c=LoS_map[unique_LoS_status][0], 
                    label=LoS_map[unique_LoS_status][1], s=2)
    plt.scatter(bs_location[0], bs_location[1], 
                c='k', marker='x', 
                label='Basestation')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('BS %i - LoS status of users'%(bs_title_idx))
    plt.legend(framealpha=.9, loc='lower left')
    plt.xlim([user_locations[:, 0].min(), user_locations[:, 0].max()])
    plt.ylim([user_locations[:, 1].min(), user_locations[:, 1].max()])

for bs_idx in [0, 1, 2]:
    bs_location = dataset[bs_idx]['location'][0]
    LoS_status = dataset[bs_idx]['user']['LoS']
    user_location = dataset[bs_idx]['user']['location']
    plot_LoS_status(bs_location, user_location, LoS_status, bs_title_idx=bs_idx+1)

# %% [markdown]
# If we want to select only the users with LoS paths to BS2 to be used in ML model, we can apply the following:

# %%
bs_idx = 1
LoS_status = dataset[bs_idx]['user']['LoS']
selected_users = LoS_status == 1

bs_location = dataset[bs_idx]['location'][0]
LoS_status = dataset[bs_idx]['user']['LoS'][selected_users]
user_locations = dataset[bs_idx]['user']['location'][selected_users]
plot_LoS_status(bs_location, user_locations, LoS_status, bs_title_idx=bs_idx+1)

# %% [markdown]
# The data to be used in the model then can be expressed as

# %%
location = dataset[bs_idx]['user']['location'][selected_users] # ML model input
channels = dataset[bs_idx]['user']['channel'][selected_users] # ML model output
print(channels.shape) # num_UEs x num_RX_ant x num_TX_ant x num_OFDM_subcarriers
print(location.shape) # num_UEs x 3 (x, y, z)

# %% [markdown]
# # Applying FoV
# The newly introduced field of view property for the antennas allow basestation to limit the transmit/receive angle of the paths. For example, for a ULA, it is more reasonable to only use the channels through the halfspace the antenna is directed at. For this, we change the TX antenna FoV parameter to (180, 180), which takes 180 degrees horizontal FOV with 180 degrees vertical FOV.
# 
# **Note:** *The parameters need to be re-initiated for a generation. The parameters after generation of a dataset is changed based on the generation of the dataset, showing additional information based on the scenario.*

# %%
parameters = DeepMIMOv3.default_params()
parameters['dataset_folder'] = r'C:\Users\Umt\Desktop\deepverse_scenarios'
parameters['scenario'] = 'city_4_phoenix'
parameters['active_BS'] = np.array([2]) # Only generate the data for the second basestation
parameters['user_rows'] = np.arange(79)
parameters['ue_antenna']['shape'] = np.array([1, 1])
parameters['bs_antenna']['shape'] = np.array([8, 1])
parameters['bs_antenna']['FoV'] = np.array([180, 180])
parameters['bs_antenna']['rotation'] = np.array([0, 0, 0]) # +x rotation
dataset = DeepMIMOv3.generate_data(parameters.copy())

bs_location = dataset[0]['location'][0]
LoS_status = dataset[0]['user']['LoS']
user_locations = dataset[0]['user']['location']
plot_LoS_status(bs_location, user_locations, LoS_status, bs_title_idx=2)

# %% [markdown]
# Now, we rotate the basestation 90 degrees towards +y

# %%
parameters['bs_antenna']['rotation'] = np.array([0, 0, 90]) # let's rotate it to +y
dataset = DeepMIMOv3.generate_data(parameters.copy())

bs_location = dataset[0]['location'][0]
LoS_status = dataset[0]['user']['LoS']
user_locations = dataset[0]['user']['location']
plot_LoS_status(bs_location, user_locations, LoS_status, bs_title_idx=2)

# %%
parameters['bs_antenna']['rotation'] = np.array([0, 0, 180]) # let's rotate it to +y
dataset = DeepMIMOv3.generate_data(parameters.copy())

bs_location = dataset[0]['location'][0]
LoS_status = dataset[0]['user']['LoS']
user_locations = dataset[0]['user']['location']
plot_LoS_status(bs_location, user_locations, LoS_status, bs_title_idx=2)

# %%
parameters['bs_antenna']['FoV'] = np.array([120, 120])
parameters['bs_antenna']['rotation'] = np.array([0, 0, 180]) # let's rotate it to +y
dataset = DeepMIMOv3.generate_data(parameters.copy())

bs_location = dataset[0]['location'][0]
LoS_status = dataset[0]['user']['LoS']
user_locations = dataset[0]['user']['location']
plot_LoS_status(bs_location, user_locations, LoS_status, bs_title_idx=2)

# %%
parameters['bs_antenna']['rotation'] = np.array([0, 0, -90]) # let's rotate it to +y
dataset = DeepMIMOv3.generate_data(parameters.copy())

bs_location = dataset[0]['location'][0]
LoS_status = dataset[0]['user']['LoS']
user_locations = dataset[0]['user']['location']
plot_LoS_status(bs_location, user_locations, LoS_status, bs_title_idx=2)

parameters['bs_antenna']['rotation'] = np.array([0, 20, -90]) # let's rotate it to +y
dataset = DeepMIMOv3.generate_data(parameters.copy())

bs_location = dataset[0]['location'][0]
LoS_status = dataset[0]['user']['LoS']
user_locations = dataset[0]['user']['location']
plot_LoS_status(bs_location, user_locations, LoS_status, bs_title_idx=2)

# %%


# %%



