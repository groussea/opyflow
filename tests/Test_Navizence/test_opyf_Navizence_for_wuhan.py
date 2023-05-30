#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 12:40:10 2019
This python file treat a video sample that has been extracte 
from a longer video taken at a higher frame rate.


@author: Gauthier
"""
#%%
# %matplotlib qt5
import opyf
import matplotlib.pyplot as plt
import sys
import os
os.chdir("./")
# if opyf is not installed where is the opyf folder?
sys.path.append('../../')
#On ipython try the magic command "%matplotlib qt5" for external outputs or "%matplotlib inline" for inline outputs

#%%

plt.close('all')


#Path toward the video file
filePath = './2018.07.04_Station_fixe_30m_sample.mp4'
#set the object information
video = opyf.videoAnalyzer(filePath)
'''
this manipualtion create an object [video] that contains information deduced from the video file.
#if it is a frame sequence use: {opyf.frameSequenceAnalyzer(path)} and type the "path" where images are.
'''

#%% ######################


video.set_vecTime(Ntot=10, shift=1, step=2, starting_frame=20)
print(video.vec, '\n', video.prev)
"""
#Use .set_vecTime vector to define the processing plan 
#This method define video.vec and video.prev, two vecotrs required for the image processing:

# (by default {.set_vecttTime(starting_frame=0, step=1, shift=1, Ntot=1)} 
to process the first two image of the video or the frame sequence and extract the velocities between these two images and produce:
#video.vec    =   [   0 ,  1 ]
#video.prev   =   [False,True]
#video.prev=False indicates that no previous image has been processed
#For video.prev=False the step consists to read the corresponding image index 
#in video.vec and extract Good Feature To Track, for True the flow will measure 
#with the pair {False+True} from the good features detected in the first image
# 
# [Ntot] specifies the total number of image pairs
# [shift] specifies the shift between two pairs
# [starting_frame] specifies the first image
# [step] specifies the number of image between 2 images of each pair. 
# WARNING: if the step increases, the displacements necessarily increase
# Note that, if the object is build from a video with videoAnalyzer, a lag is expected since each 
    required images for the process are loaded in the memory for efficiency reasons
# 
# This function also defines video.Time, that is the time vector at which the velocity measurements are performed
# =============================================================================
"""
#%%
video.extractGoodFeaturesAndDisplacements(
    display='quiver', displayColor=True, width=0.002)
'''
#the method {.extractGoodFeaturesAndDisplacements} applied to the object video will detect the good feature to track and calculate the optical flow according 
#to the processing plan defined by set_vecTime. The option 'quiver' display the velocity vectors corresponding to the feature to track, while display='points'
#shows the positions only. displayColor introduce a colormap that correspond to the velocitiy magnitude. you can use the usual arguments for plotting with [plt.quiver] for quiver or [plt.scatter] for the points
'''

#%%

video.set_vlim([0, 30])
'''
#set_vlim defines video.vlim and indicates the range of displacement expected with the processing plan (close link with step parameter in set_vecTime) 
#you can run again the processing and see the difference with above

'''

video.extractGoodFeaturesAndDisplacements(
    display='quiver', displayColor=True, width=0.002)
#%%
video.set_filtersParams(wayBackGoodFlag=4, RadiusF=20,
                        maxDevInRadius=1, CLAHE=True)
'''
# =============================================================================
# Now you may want to apply some filters to erase outliers
# wayBackGoodFlag] specify the distance accepted between the initial feature position and the 
# position calculated by using the displacement 
# rom A to B and then from B to A (where A and B are the pair on wich optical flow is calculated)
# CLAHE For Contrast Limited Adaptative Histogram Equalization. for https://en.wikipedia.org/wiki/Adaptive_histogram_equalization#Contrast_Limited_AHE
# Enhamces local contrasts and  indicates if you want to well-balanced (may improve considerably the results or (in opposite) introduce unwanted bias)
# RadiusF and maxDevInRadius specify the maxiimum deviation expected within RadiusF. 
#Deviation is calculated with Dev=(x-mean{S})/std(S), where s is the velocity values within RadiusF. x is the value of interest on wich we test the filter. if Dev>maxDev the value is deleted
'''

#%%%%%%%%%%%%%%%%%%%%%%%%%
'''
#You can specify the goodFeatureToTracks params (more information on https://docs.opencv.org/master/d4/d8c/tutorial_py_shi_tomasi.html or 
#Shi, Jianbo. "Good features to track." 1994 Proceedings of IEEE conference on computer vision and pattern recognition. IEEE, 1994.)
'''
video.set_goodFeaturesToTrackParams(maxCorners=50000, qualityLevel=0.001)
#access to these parameters with:
video.feature_params

#%%#######################
'''
#you may specify the opticalFlow parameters 
#default is video.set_opticalFlowParams(winSize=(16, 16), maxLevel=3)
#more information on https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html
'''
video.set_opticalFlowParams(maxLevel=3)
#and access
video.lk_params


#%%


video.extractGoodFeaturesPositionsDisplacementsAndInterpolate(
    display='field', displayColor=True, scale=80, width=0.005)
'''
#Extract the velocity field by interpolating the displacements of the 'goodFeaturesToTrack' on a field defined by the method {set_gridToInterpolateOn}. 
#By default, this grid is set at (pixLeft=0, pixRight=0, stepHor=2, pixUp=0, pixDown=0, stepVert=2)

#Interpolated datas with accumulation are stored in video.UxTot[k] and video.UyTot[k], for velocities at video.Time[k]
'''

#%%
# To increase the quality you may increase the number of pars of frames treated Ntot. 
# Here we set Ntot at 88 instead of 10 in the first example.
video.set_vecTime(starting_frame=10,step=2,shift=1,Ntot=88)
# To obtain a convergence of flow statistics we should collect velocity at 
# least for 3 secondd

video.set_interpolationParams(Sharpness=2,Radius=40)

video.extractGoodFeaturesDisplacementsAccumulateAndInterpolate(
    display1='quiver', display2='field', displayColor=True, scale=200)
'''
#If the method {extractGoodFeaturesDisplacementsAccumulateAndInterpolate} is applied, only one field will be produced at the end of the processing
#display 1='quiver' displays the GFT at each time step
#diplay 2='field' displays the finale field interpolated on the grid from all the velocity vectors
#This method is regularly employed when flow is expected to be permanent or quasi static during an interval that contains several frames.
#it is also appropriate to obtain the average velocity field from a sequence

#Interpolated datas with accumulation are stored in video.UxTot[0] and video.UyTot[0]


'''


#%%
'''
To extract field data use
'''
video.writeVelocityField(fileFormat='csv')
'''
#for csv format
#if {extractGoodFeaturesDisplacementsAccumulateAndInterpolate} is run. only one file is generated for the field resulting from the accumulation of vectors
#if {extractGoodFeaturesPositionsDisplacementsAndInterpolate}  is run, a serie of csv files is generated

#The format is :
#   X, Y, Ux, Uy

#be aware that by default the vertical axis Y is  oriented downward for images
'''
'''
#To save data in smaller files the most convenient format is hdf5
'''
video.writeVelocityField(fileFormat='hdf5')
#check if the file is readable
opyf.hdf5_Read(video.filename+'.hdf5')


#%% ##############3


video.scaleData(framesPerSecond=25, metersPerPx=0.02,
                unit=['m', 's'], origin=[0, video.Hvis])

'''
# the scaling function if you want to scale data, i.e., give the fps and the length scale
method is applied, there is no possibility to go back to Unscale, however it is possible to continue to process with scaling

Here the scale is 2 cemtimeters per px.
#The Y axis is now oriented upward
#check that with 
'''
video.showXV(video.X, video.V, display='points', displayColor=True)
#or the averaged velocity field

Field = opyf.Render.setField(video.UxTot[0], video.UyTot[0], 'norm')
video.opyfDisp.plotField(Field, vis=video.vis)


video.interpolateOnGrid(video.Xaccu, video.Vaccu)
import numpy as np
Ux=np.reshape( video.interpolatedVelocities[:, 0], (video.Hgrid, video.Lgrid))
plt.plot(Ux[:,40])


#%%

'''
for plotting only the resulting averaged field, usefull if Ntot is longer
'''
video.set_vlim([0, 30])
video.set_vecTime(Ntot=10, shift=1, step=1, starting_frame=20)
video.extractGoodFeaturesDisplacementsAccumulateAndInterpolate(
    display2='field', displayColor=True, scale=200)


#%%
video.set_trackingFeatures(
    Ntot=10, step=1, starting_frame=1, track_length=5, detection_interval=10)
'''
# the method {extractTracks} is available to extract tracks on images. The principle is inspired by the openCv sample lktrack.py (https://github.com/opencv/opencv/blob/master/samples/python/lk_track.py)
#The main difference rely on the possibility to store the tracks 
#Note that, if the object is build from a video with videoAnalyzer, a lag is expected since each required images a loaded in the memory for efficiency reasons
It might be a technic to better filter relevant velocities since it is possible to follow these patterns for multiple frames 
'''
opyf.mkdir2('./export_Tracks/')
video.set_filtersParams(wayBackGoodFlag=1, CLAHE=False)
video.extractTracks(display='quiver', displayColor=True,
                    saveImgPath='./export_Tracks/', numberingOutput=True)
# tracks can be saved in csv file
video.writeTracks(outFolder='./export_Tracks')
#when wrtiting it is possible to specify the out folder
