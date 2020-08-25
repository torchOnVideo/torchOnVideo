"""
Video Tracking Module
----------------------------
This module does Video Tracking

"""


import torch
import numpy as np


class VideoTracking():
    '''
    Native Class for Video Tracking, acts like an interface for all other Video 
    '''
    
    def __init__(self, tracking_type):
        self.TrackingType = tracking_type

    def get_models(self):
        raise NotImplementedError
