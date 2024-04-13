# -*- coding: utf-8 -*-
"""
DeepMIMOv3 Python Implementation

Description: Utilities

Authors: Umut Demirhan, Ahmed Alkhateeb
Date: 12/10/2021
"""

import time
import DeepMIMOv3.consts as c
from DeepMIMOv3.construct_deepmimo import ant_indices, array_response
import numpy as np

# Sleep between print and tqdm displays
def safe_print(text, stop_dur=0.3):
    print(text)
    time.sleep(stop_dur)
        

# Determine active paths with the given configurations
# (For OFDM, only the paths within DS are activated)
class PathVerifier:
    def __init__(self, params):
        self.params = params
        if self.params[c.PARAMSET_FDTD]: # IF OFDM
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
        if self.params[c.PARAMSET_FDTD]: # IF OFDM
            avg_ratio_FFT = 0
            if len(self.path_ratio_FFT) != 0:
                avg_ratio_FFT = np.mean(self.path_ratio_FFT)*100
                
            if self.max_ToA > self.FFT_duration and avg_ratio_FFT >= 1.:
                safe_print('ToA of some paths of %i channels with an average total power of %.2f%% exceed the useful OFDM symbol duration and are clipped.' % (len(self.path_ratio_FFT), avg_ratio_FFT))
            
def dbm2pow(val):
    return 10**(val/10 - 3)

def steering_vec(array, phi=0, theta=0, kd=np.pi):
    # phi = azimuth
    # theta = elevation
    idxs = ant_indices(array)
    resp = array_response(idxs, phi, theta+np.pi/2, kd)
    return resp / np.linalg.norm(resp)
