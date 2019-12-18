#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 08:24:12 2017

@author: Gauthier ROUSSEAU
"""

# WARNING : to work properly you must run this script from 'Test_land_slide_youtube_video' FOlder

import sys
import os
sys.path.append('../../')
import opyf
import json
import matplotlib.pyplot as plt
import numpy as np
import cv2
from pytube import YouTube
folder_main = '.'
os.chdir(folder_main)


plt.rc('text', usetex=False)

# %%

# command lines To save the online youtube  video of the land slide (need the pytube package)
# 'pip install pytube' on the cmd prompt

# You may run this script for any video link from youtube
yt = YouTube("https://www.youtube.com/watch?time_continue=2&v=5-nyAz484WA")
stre = yt.streams.first()
# stre.download(folder_main)

# %%
# if the video has been correctly downloaded you may run this commands to load the video name.

vidName = stre.default_filename

vidPath = folder_main+'/'+vidName

landSlide=opyf.videoAnalyzer(vidPath)

#%%
landSlide.set_vlim([0,20])
landSlide.set_vecTime(Ntot=10,shift=2)
landSlide.set_filtersParams(maxDevInRadius=1)
landSlide.extractGoodFeaturesAndDisplacements()

#%% And now to draw a FIeld

landSlide.extractGoodFeaturesPositionsDisplacementsAndInterpolate()



#%% optional by loading the mask


landSlide=opyf.videoAnalyzer(vidPath,mask=folder_main+'/mask.png')
landSlide.extractGoodFeaturesPositionsDisplacementsAndInterpolate()

