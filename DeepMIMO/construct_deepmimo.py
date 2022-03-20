# -*- coding: utf-8 -*-
"""
DeepMIMOv2 Python Implementation

Description: MIMO channel generator

Authors: Umut Demirhan, Ahmed Alkhateeb
Date: 12/10/2021
"""

import DeepMIMO.consts as c
import numpy as np
from tqdm import tqdm
import types
from DeepMIMO.ant_patterns import AntennaPattern

# Generates common parameters first. The output is a numpy matrix.
def generate_MIMO_channel(raydata, params, tx_ant_params, rx_ant_params):
    
    bandwidth = params[c.PARAMSET_OFDM][c.PARAMSET_OFDM_BW] * c.PARAMSET_OFDM_BW_MULT
    
    kd_tx = 2*np.pi*tx_ant_params[c.PARAMSET_ANT_SPACING]
    kd_rx = 2*np.pi*rx_ant_params[c.PARAMSET_ANT_SPACING]
    Ts = 1/bandwidth
    subcarriers = np.arange(0, params[c.PARAMSET_OFDM][c.PARAMSET_OFDM_SC_LIM], params[c.PARAMSET_OFDM][c.PARAMSET_OFDM_SC_SAMP])
    path_gen = OFDM_PathGenerator(params[c.PARAMSET_OFDM], subcarriers)
    antennapattern = AntennaPattern(tx_pattern = tx_ant_params[c.PARAMSET_ANT_RAD_PAT], rx_pattern = rx_ant_params[c.PARAMSET_ANT_RAD_PAT])

    M_tx = np.prod(tx_ant_params[c.PARAMSET_ANT_SHAPE])
    ant_tx_ind = ant_indices(tx_ant_params[c.PARAMSET_ANT_SHAPE])
    
    M_rx = np.prod(rx_ant_params[c.PARAMSET_ANT_SHAPE])
    ant_rx_ind = ant_indices(rx_ant_params[c.PARAMSET_ANT_SHAPE])
    
    if  params[c.PARAMSET_FDTD]:
        channel = np.zeros((len(raydata), M_rx, M_tx, len(subcarriers)), dtype = np.csingle)
    else:
        channel = np.zeros((len(raydata), M_rx, M_tx, params[c.PARAMSET_NUM_PATHS]), dtype = np.csingle)
        
    for i in tqdm(range(len(raydata)), desc='Generating channels'):
        
        if raydata[i][c.OUT_PATH_NUM]==0:
            continue
        
        
        dod_theta, dod_phi = rotate_angles(rotation = tx_ant_params[c.PARAMSET_ANT_ROTATION],
                                  theta = raydata[i][c.OUT_PATH_DOD_THETA],
                                  phi = raydata[i][c.OUT_PATH_DOD_PHI])
        
        doa_theta, doa_phi = rotate_angles(rotation = rx_ant_params[c.PARAMSET_ANT_ROTATION][i],
                                  theta = raydata[i][c.OUT_PATH_DOA_THETA],
                                  phi = raydata[i][c.OUT_PATH_DOA_PHI])
        
        array_response_TX = array_response(ant_ind = ant_tx_ind, 
                                           theta = dod_theta, 
                                           phi = dod_phi, 
                                           kd = kd_tx)
        
        array_response_RX = array_response(ant_ind = ant_rx_ind, 
                                           theta =  doa_theta, 
                                           phi = doa_phi,
                                           kd = kd_rx)
        
        power = antennapattern.apply(power = raydata[i][c.OUT_PATH_RX_POW], 
                                     doa_theta = doa_theta, 
                                     doa_phi = doa_phi, 
                                     dod_theta = dod_theta, 
                                     dod_phi = dod_phi)
        
        if  params[c.PARAMSET_FDTD]: # OFDM
            path_const = path_gen.generate(power.reshape(-1, 1), 
                                           (raydata[i][c.OUT_PATH_TOA]/Ts).reshape(-1, 1), 
                                           raydata[i][c.OUT_PATH_PHASE].reshape(-1,1))
            
            # The next step is to be defined
            if params[c.PARAMSET_OFDM][c.PARAMSET_OFDM_LPF] == 0:
                channel[i] = np.sum(array_response_RX[:, None, None, :] * array_response_TX[None, :, None, :] * path_const.T[None, None, :, :], axis=3)
            else:
                channel[i] = np.sum(array_response_RX[:, None, None, :] * array_response_TX[None, :, None, :] * path_const.T[None, None, :, :], axis=3) @ path_gen.delay_to_OFDM
        
        else: # TD channel
            channel[i, :, :, :raydata[i][c.OUT_PATH_NUM]] = array_response_RX[:, None, :] * array_response_TX[None, :, :] * (np.sqrt(power) * np.exp(1j*np.deg2rad(raydata[i][c.OUT_PATH_PHASE])))[None, None, :]

    return channel

# Generates everything for each rx_ant_params - 
# necessary for BS-BS channels since different antennas can be defined. 
# The output is a list.
def generate_MIMO_channel_rx_ind(raydata, params, tx_ant_params, rx_ant_params):
    
    bandwidth = params[c.PARAMSET_OFDM][c.PARAMSET_OFDM_BW] * c.PARAMSET_OFDM_BW_MULT
    
    Ts = 1/bandwidth
    subcarriers = np.arange(0, params[c.PARAMSET_OFDM][c.PARAMSET_OFDM_SC_LIM], params[c.PARAMSET_OFDM][c.PARAMSET_OFDM_SC_SAMP])
    path_gen = OFDM_PathGenerator(params[c.PARAMSET_OFDM], subcarriers)
    
    
    channel = []
        
    for i in tqdm(range(len(raydata)), desc='Generating channels'):
        
        antennapattern = AntennaPattern(tx_pattern = tx_ant_params[c.PARAMSET_ANT_RAD_PAT], rx_pattern = rx_ant_params[i][c.PARAMSET_ANT_RAD_PAT])
        
        kd_tx = 2*np.pi*tx_ant_params[c.PARAMSET_ANT_SPACING]
        kd_rx = 2*np.pi*rx_ant_params[i][c.PARAMSET_ANT_SPACING]
        
        M_tx = np.prod(tx_ant_params[c.PARAMSET_ANT_SHAPE])
        ant_tx_ind = ant_indices(tx_ant_params[c.PARAMSET_ANT_SHAPE])
        
        M_rx = np.prod(rx_ant_params[i][c.PARAMSET_ANT_SHAPE])
        ant_rx_ind = ant_indices(rx_ant_params[i][c.PARAMSET_ANT_SHAPE])
        
        if raydata[i][c.OUT_PATH_NUM]==0:
            channel.append(np.zeros((M_rx, M_tx, len(subcarriers))))
            continue
    
        
        dod_theta, dod_phi = rotate_angles(rotation = tx_ant_params[c.PARAMSET_ANT_ROTATION],
                                  theta = raydata[i][c.OUT_PATH_DOD_THETA],
                                  phi = raydata[i][c.OUT_PATH_DOD_PHI])
        
        doa_theta, doa_phi = rotate_angles(rotation = rx_ant_params[i][c.PARAMSET_ANT_ROTATION],
                                  theta = raydata[i][c.OUT_PATH_DOA_THETA],
                                  phi = raydata[i][c.OUT_PATH_DOA_PHI])
        
        array_response_TX = array_response(ant_ind = ant_tx_ind, 
                                           theta = dod_theta, 
                                           phi = dod_phi, 
                                           kd = kd_tx)
        
        array_response_RX = array_response(ant_ind = ant_rx_ind, 
                                           theta =  doa_theta, 
                                           phi = doa_phi,
                                           kd = kd_rx)
        
        power = antennapattern.apply(power = raydata[i][c.OUT_PATH_RX_POW], 
                                     doa_theta = doa_theta, 
                                     doa_phi = doa_phi, 
                                     dod_theta = dod_theta, 
                                     dod_phi = dod_phi)
        
        if  params[c.PARAMSET_FDTD]: # OFDM
            path_const = path_gen.generate(power.reshape(-1, 1), 
                                           (raydata[i][c.OUT_PATH_TOA]/Ts).reshape(-1, 1), 
                                           raydata[i][c.OUT_PATH_PHASE].reshape(-1,1))
            
            # The next step is to be defined
            if params[c.PARAMSET_OFDM][c.PARAMSET_OFDM_LPF] == 0: # NO LPF
                channel.append(np.sum(array_response_RX[:, None, None, :] * array_response_TX[None, :, None, :] * path_const.T[None, None, :, :], axis=3))
            else: # Sinc LPF
                channel.append(np.sum(array_response_RX[:, None, None, :] * array_response_TX[None, :, None, :] * path_const.T[None, None, :, :], axis=3) @ path_gen.delay_to_OFDM)
        
        else: # Time domain
            channel.append(array_response_RX[:, None, :] * array_response_TX[None, :, :] * (np.sqrt(power) * np.exp(1j*np.deg2rad(raydata[i][c.OUT_PATH_PHASE])))[None, None, :])

    return channel


def array_response(ant_ind, theta, phi, kd):        
    gamma = array_response_phase(theta, phi, kd)
    return np.exp(ant_ind@gamma.T)
    
def array_response_phase(theta, phi, kd):
    gamma_x = 1j*kd*np.sin(theta)*np.cos(phi)
    gamma_y = 1j*kd*np.sin(theta)*np.sin(phi)
    gamma_z = 1j*kd*np.cos(theta)
    return np.vstack([gamma_x, gamma_y, gamma_z]).T
 
def ant_indices(panel_size):
    gamma_x = np.tile(np.arange(panel_size[0]), panel_size[1]*panel_size[2])
    gamma_y = np.tile(np.repeat(np.arange(panel_size[1]), panel_size[0]), panel_size[2])
    gamma_z = np.repeat(np.arange(panel_size[2]), panel_size[0]*panel_size[1])
    return np.vstack([gamma_x, gamma_y, gamma_z]).T

def rotate_angles(rotation, theta, phi): # Input all degrees - output radians
    theta = np.deg2rad(theta)
    phi = np.deg2rad(phi)

    if rotation is not None:
        rotation = np.deg2rad(rotation)
    
        sin_alpha = np.sin(phi - rotation[2])
        sin_beta = np.sin(rotation[1])
        sin_gamma = np.sin(rotation[0])
        cos_alpha = np.cos(phi - rotation[2])
        cos_beta = np.cos(rotation[1])
        cos_gamma = np.cos(rotation[0])
        
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        
        theta = np.arccos(cos_beta*cos_gamma*cos_theta 
                              + sin_theta*(sin_beta*cos_gamma*cos_alpha-sin_gamma*sin_alpha)
                              )
        phi = np.angle(cos_beta*sin_theta*cos_alpha-sin_beta*cos_theta 
                           + 1j*(cos_beta*sin_gamma*cos_theta 
                                 + sin_theta*(sin_beta*sin_gamma*cos_alpha + cos_gamma*sin_alpha))
                           )
    return theta, phi

class OFDM_PathGenerator:
    def __init__(self, OFDM_params, subcarriers):
        self.OFDM_params = OFDM_params
        
        if OFDM_params[c.PARAMSET_OFDM_LPF] == 0: # No Pulse Shaping
            self.generate = getattr(self, 'no_LPF')
        else: # Pulse Shaping
            self.generate = getattr(self, 'with_LPF')
            
                
        self.subcarriers = subcarriers
        self.total_subcarriers = OFDM_params[c.PARAMSET_OFDM_SC_NUM]
        
        self.delay_d = np.arange(self.OFDM_params['subcarriers'])
        self.delay_to_OFDM = np.exp(-1j*2*np.pi/self.total_subcarriers*np.outer(self.delay_d, self.subcarriers))
        
    def no_LPF(self, power, delay_n, phase):
        paths_over_FFT = (delay_n >= self.OFDM_params['subcarriers'])
        power[paths_over_FFT] = 0
        delay_n[paths_over_FFT] = self.OFDM_params['subcarriers']
        
        return np.sqrt(power/self.total_subcarriers) * np.exp(1j*(np.deg2rad(phase) - (2*np.pi/self.total_subcarriers)*np.outer(delay_n, self.subcarriers) ))
    
    def with_LPF(self, power, delay_n, phase):
        
        # Ignore the paths over CP
        paths_over_FFT = (delay_n >= self.OFDM_params['subcarriers'])
        power[paths_over_FFT] = 0
        delay_n[paths_over_FFT] = self.OFDM_params['subcarriers']
        
        # Pulse - LPF convolution and channel generation
        pulse = np.sinc(self.delay_d-delay_n)
        pulse = pulse * np.sqrt(power/self.total_subcarriers) * np.exp(1j*np.deg2rad(phase)) # Power scaling
        return pulse
    