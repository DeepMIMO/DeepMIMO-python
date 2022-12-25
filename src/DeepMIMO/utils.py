# -*- coding: utf-8 -*-
"""
DeepMIMOv2 Python Implementation

Description: Utilities

Authors: Umut Demirhan, Ahmed Alkhateeb
Date: 12/10/2021
"""

import time

# Sleep between print and tqdm displays
def safe_print(text, stop_dur=0.3):
    print(text)
    time.sleep(stop_dur)