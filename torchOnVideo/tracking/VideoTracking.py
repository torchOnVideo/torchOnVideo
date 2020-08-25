"""
Video Tracking Module
----------------------------
This module does Video Tracking

"""


import torch
import numpy as np


class VideoTracking():
    
    def __init__(self, tracking_type):
        self.TrackingType = tracking_type

    def get_models(self):
        raise NotImplementedError
