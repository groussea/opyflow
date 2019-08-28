#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 16:55:20 2019

@author: gauthier
"""

import numpy as np
import cv2


def convertToGrayScale(vis):
    if len(np.shape(vis)) == 3:
        gray = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)
    else:
        gray = vis
    return gray
