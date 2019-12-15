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

def computeHomography(R1, tvec1, R2,tvec2,d_inv, normal):
    homography = R2 @ R1.transpose() + d_inv * (-R2 @ R1.transpose() @ tvec1 + tvec2) @ normal.transpose()
    return homography