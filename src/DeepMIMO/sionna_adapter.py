# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 10:53:50 2022

@author: Umut Demirhan, Ahmed Alkhateeb
"""
import numpy as np

# The DeepMIMO dataset adapter for Sionna
#
# Input: DeepMIMO dataset, UE and BS indices to be included
#
# For a given 1D vector of BS or UE indices, the generated dataset will be stacked as different samples
#
# By default, the adapter will only select the first BS of the DeepMIMO dataset and all UEs
# The adapter assumes BSs are transmitters and users are receivers. 
# Uplink channels could be generated using (transpose) reciprocity.
#
# For multi-user channels, provide a 2D numpy matrix of size (num_samples x num_rx)
#
# Examples:
# ue_idx = np.array([[0, 1 ,2], [1, 2, 3]]) generates (num_bs x 3 UEs) channels
# with 2 data samples from the BSs to the UEs [0, 1, 2] and [1, 2, 3], respectively.
#
# For single-basestation channels with the data from different basestations stacked,
# provide a 1D array of basestation indices
#
# For multi-BS channels, provide a 2D numpy matrix of (num_samples x num_tx)
#
# Examples:
# bs_idx = np.array([[0, 1], [2, 3], [4, 5]]) generates (2 BSs x num_rx) channels
# by stacking the data of channels from the basestations (0 and 1), (2 and 3), 
# and (4 and 5) to the UEs.
#
class DeepMIMOSionnaAdapter:
    def __init__(self, DeepMIMO_dataset, bs_idx = None, ue_idx = None):
        self.dataset = DeepMIMO_dataset
        
        # Set bs_idx based on given parameters
        # If no input is given, choose the first basestation
        if bs_idx is None:
            bs_idx = np.array([[0]])
        self.bs_idx = self._verify_idx(bs_idx)
        
        # Set ue_idx based on given parameters
        # If no input is given, set all user indices
        if ue_idx is None:
            ue_idx = np.arange(DeepMIMO_dataset[0]['user']['channel'].shape[0])
        self.ue_idx = self._verify_idx(ue_idx)
        
        # Extract number of antennas from the DeepMIMO dataset
        self.num_rx_ant = DeepMIMO_dataset[0]['user']['channel'].shape[1]
        self.num_tx_ant = DeepMIMO_dataset[0]['user']['channel'].shape[2]
        
        # Determine the number of samples based on the given indices
        self.num_samples_bs = self.bs_idx.shape[0]
        self.num_samples_ue = self.ue_idx.shape[0]
        self.num_samples = self.num_samples_bs * self.num_samples_ue
        
        # Determine the number of tx and rx elements in each channel sample based on the given indices
        self.num_rx = self.ue_idx.shape[1]
        self.num_tx = self.bs_idx.shape[1]
        
        # Determine the number of available paths in the DeepMIMO dataset
        self.num_paths = DeepMIMO_dataset[0]['user']['channel'].shape[-1]
        self.num_time_steps = 1 # Time step = 1 for static scenarios
        
        # The required path power shape for Sionna
        self.ch_shape = (self.num_rx, 
                         self.num_rx_ant, 
                         self.num_tx, 
                         self.num_tx_ant, 
                         self.num_paths, 
                         self.num_time_steps)
        
        # The required path delay shape for Sionna
        self.t_shape = (self.num_rx, self.num_tx, self.num_paths)
    
    # Verify the index values given as input
    def _verify_idx(self, idx):
        idx = self._idx_to_numpy(idx)
        idx = self._numpy_size_check(idx)
        return idx
    
    # Convert the possible input types to numpy (integer - range - list)
    def _idx_to_numpy(self, idx):
        if isinstance(idx, int): # If the input is an integer - a single ID - convert it to 2D numpy array
            idx = np.array([[idx]])
        elif isinstance(idx, list) or isinstance(idx, range): # If the input is a list or range - convert it to a numpy array
            idx = np.array(idx)
        elif isinstance(idx, np.ndarray):
            pass
        else:
            raise TypeError('The index input type must be an integer, list, or numpy array!') 
        return idx
    
    # Check the size of the given input and convert it to a 2D matrix of proper shape (num_tx x num_samples) or (num_rx x num_samples)
    def _numpy_size_check(self, idx):
        if len(idx.shape) == 1:
            idx = idx.reshape((-1, 1))
        elif len(idx.shape) == 2:
            pass
        else:
            raise ValueError('The index input must be integer, vector or 2D matrix!')
        return idx
    
    # Override length of the generator to provide the available number of samples
    def __len__(self):
        return self.num_samples
        
    # Provide samples each time the generator is called
    def __call__(self):
        for i in range(self.num_samples_ue): # For each UE sample
            for j in range(self.num_samples_bs): # For each BS sample
                # Generate zero vectors for the Sionna sample
                a = np.zeros(self.ch_shape, dtype=np.csingle)
                tau = np.zeros(self.t_shape, dtype=np.single)
                
                # Place the DeepMIMO dataset power and delays into the channel sample for Sionna
                for i_ch in range(self.num_rx): # for each receiver in the sample
                    for j_ch in range(self.num_tx): # for each transmitter in the sample
                        i_ue = self.ue_idx[i][i_ch] # UE channel sample i - channel RX i_ch
                        i_bs = self.bs_idx[j][j_ch] # BS channel sample i - channel TX j_ch
                        a[i_ch, :, j_ch, :, :, 0] = self.dataset[i_bs]['user']['channel'][i_ue]
                        tau[i_ch, j_ch, :self.dataset[i_bs]['user']['paths'][i_ue]['num_paths']] = self.dataset[i_bs]['user']['paths'][i_ue]['ToA'] 
                
                yield (a, tau) # yield this sample
