# -*- coding: utf-8 -*-
"""
DeepMIMOv2 Python Implementation

Description: Antenna Radiation Patterns

Authors: Umut Demirhan, Ahmed Alkhateeb
Date: 3/16/2022
"""

import DeepMIMO.consts as c
import numpy as np

class AntennaPattern():
    def __init__(self, tx_pattern, rx_pattern):
        # Initialize TX Pattern
        if tx_pattern in c.PARAMSET_ANT_RAD_PAT_VALS:
            if tx_pattern == c.PARAMSET_ANT_RAD_PAT_VALS[0]:
                self.tx_pattern_fn = None
            else:
                tx_pattern = tx_pattern.replace('-', '_')
                tx_pattern = 'pattern_' + tx_pattern
                self.tx_pattern_fn = globals()[tx_pattern]
        else:
            raise NotImplementedError('The given \'%s\' antenna radiation pattern is not applicable.' % tx_pattern)
        
        
        # Initialize RX Pattern
        if rx_pattern in c.PARAMSET_ANT_RAD_PAT_VALS:
            if rx_pattern == c.PARAMSET_ANT_RAD_PAT_VALS[0]:
                self.rx_pattern_fn = None
            else:
                rx_pattern = rx_pattern.replace('-', '_')
                rx_pattern = 'pattern_' + rx_pattern
                self.rx_pattern_fn = globals()[rx_pattern]
        else:
            raise NotImplementedError('The given \'%s\' antenna radiation pattern is not applicable.' % rx_pattern)
    
    def apply(self, power, doa_theta, doa_phi, dod_theta, dod_phi):
        pattern = 1.
        if self.tx_pattern_fn is not None:
            pattern *= self.tx_pattern_fn(dod_theta, dod_phi)
        if self.rx_pattern_fn is not None:
            pattern *= self.rx_pattern_fn(doa_theta, doa_phi)
            
        return power * pattern

def pattern_halfwave_dipole(theta, phi):
    max_gain = 1.6409223769 # Half-wave dipole maximum directivity
    theta_nonzero = theta.copy()
    zero_idx = theta_nonzero==0
    theta_nonzero[zero_idx] = 1e-4 # Approximation of 0 at limit
    pattern = max_gain * np.cos((np.pi/2)*np.cos(theta))**2 / np.sin(theta)**2
    pattern[zero_idx] = 0
    return pattern
    
