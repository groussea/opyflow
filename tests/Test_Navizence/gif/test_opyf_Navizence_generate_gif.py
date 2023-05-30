#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 12:40:10 2019

@author: Gauthier
"""
#%%

import sys, os
os.chdir("./")
# if opyf is not installed where is the opyf folder?
sys.path.append('../../')
import opyf 
import matplotlib.pyplot as plt
#On ipython try the magic command "%matplotlib qt5" for external outputs or "%matplotlib inline" for inline outputs

%matplotlib qt5

plt.close('all')


#Path toward the video file
filePath='./2018.07.04_Station_fixe_30m_sample.mp4'

filePath='/media/gauthier/Data-Gauthier/programs/gitHub/opyflow/test/Test_Navizence//2018.07.04_Station_fixe_30m_sample.mp4'

#set the object information
video=opyf.videoAnalyzer(filePath)

#%%
import numpy as np
video.Field = np.zeros_like(video.grid_x)
video.opyfDisp.ax.set_position([0.12232868757259015,
0.24062988281250006,
0.7953426248548199,
0.6687402343749999])
video.opyfDisp.ax.set_aspect('equal', adjustable='box')
video.opyfDisp.fig.set_size_inches(7, 4)
video.opyfDisp.ax.set_xlabel('')
video.Field[np.where(video.Field==0)]=np.nan
video.scaleData(framesPerSecond=25, metersPerPx=12.1/600, unit=['m', 's'], origin=[0,video.Hvis])
video.opyfDisp.plotField(video.Field, vis=video.vis, vlim=[0, 5])
video.opyfDisp.fig.savefig('/media/gauthier/Data-Gauthier/programs/gitHub/opyflow/test/Test_Navizence/gif/init.png', dpi=142)

#%% Generate intermediate frames

for s in range(20,55,5):
    video.set_vecTime(Ntot=2, step=2, shift=1, starting_frame=s)
    video.set_goodFeaturesToTrackParams(maxCorners=50000, qualityLevel=0.001)
    video.set_filtersParams(wayBackGoodFlag=4,RadiusF=2,maxDevInRadius=1,CLAHE=True)
    video.scaleData(framesPerSecond=25, metersPerPx=12.1/600, unit=['m', 's'], origin=[0,video.Hvis])
    video.set_vlim([0,5])
    video.extractGoodFeaturesDisplacementsAndAccumulate()
    # video.opyfDisp.plotField(video.Field,vlim=[0,60])
    video.opyfDisp.ax.set_position([0.12232868757259015,
    0.24062988281250006,
    0.7953426248548199,
    0.6687402343749999])
    video.opyfDisp.ax.set_aspect('equal', adjustable='box')
    video.opyfDisp.fig.set_size_inches(7, 4)
    video.opyfDisp.ax.set_xlabel('')
    # video.invertYaxis()

    video.showXV(video.Xaccu, video.Vaccu, vis=video.vis, display='quiver',nvec=5000,vlim=[0,5])
    video.opyfDisp.fig.savefig('/media/gauthier/Data-Gauthier/programs/gitHub/opyflow/test/Test_Navizence/gif/frame'+str(s)+'.png', dpi=142)



#%%


video.set_vecTime(Ntot=20, shift=2, step=2, starting_frame=20)
video.set_filtersParams(wayBackGoodFlag=4,RadiusF=2,maxDevInRadius=1,CLAHE=True)
video.set_goodFeaturesToTrackParams(maxCorners=50000,qualityLevel=0.001)
video.scaleData(framesPerSecond=25, metersPerPx=12.1/600, unit=['m', 's'], origin=[0,video.Hvis])
video.set_vlim([0,5])
video.extractGoodFeaturesDisplacementsAccumulateAndInterpolate(display2='field',displayColor=True,scale=200)

# video.showXV(video.X, video.V,vis=video.vis,display='points',displayColor=True)
#or the averaged velocity field
video.set_interpolationParams(Sharpness=4)
video.interpolateOnGrid(video.Xaccu,video.Vaccu)


# Field = opyf.Render.setField(video.UxTot[0], video.UyTot[0], 'norm')
# video.opyfDisp.plotField(Field, vis=video.vis)

#%% filtering final results

video.set_filtersParams(RadiusF=5*,maxDevInRadius=2,minNperRadius=3)
video.filterAndInterpolate()

#%% Rendering to have the good format for the gif
video.opyfDisp.ax.set_position([0.12232868757259015,
 0.24062988281250006,
 0.7953426248548199,
 0.6687402343749999])

video.opyfDisp.ax.set_aspect('equal', adjustable='box')
video.opyfDisp.fig.set_size_inches(7, 4)
video.opyfDisp.ax.set_xlabel('')
plt.show()
video.opyfDisp.fig.savefig('frame_final.png', dpi=142)

#%%

video.extractGoodFeaturesAndDisplacements(display='quiver',displayColor=True,width=0.002)
'''
#the method {.extractGoodFeaturesAndDisplacements} applied to the object video will detect the good feature to track and calculate the optical flow according 
#to the processing plan defined by set_vecTime. The option 'quiver' display the velocity vectors corresponding to the feature to track, while display='points'
#shows the positions only. displayColor introduce a colormap that correspond to the velocitiy magnitude. you can use the usual arguments for plotting with [plt.quiver] for quiver or [plt.scatter] for the points
'''

#%%

video.set_vlim([0,30])
'''
#set_vlim defines video.vlim and indicates the range of displacement expected with the processing plan (close link with step parameter in set_vecTime) 
#you can run again the processing and see the difference with above

'''

video.extractGoodFeaturesAndDisplacements(display='quiver',displayColor=True,width=0.002)
#%%
video.set_filtersParams(wayBackGoodFlag=4,RadiusF=20,maxDevInRadius=1,CLAHE=True)
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
video.set_goodFeaturesToTrackParams(maxCorners=50000,qualityLevel=0.001)
#access to these parameters with:
video.feature_params

#%%#######################
'''
#you may specify the opticalFlow parameters 
#default is video.set_opticalFlowParams(winSize=(16, 16), maxLevel=3)
#more information on https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html
'''
#and access


#%% 

video.extractGoodFeaturesPositionsDisplacementsAndInterpolate(display='field',displayColor=True,scale=80,width=0.005)
'''
#Extract the velocity field by interpolating the displacements of the 'goodFeaturesToTrack' on a field defined by the method {set_gridToInterpolateOn}. 
#By default, this grid is set at (pixLeft=0, pixRight=0, stepHor=2, pixUp=0, pixDown=0, stepVert=2)

#Interpolated datas with accumulation are stored in video.UxTot[k] and video.UyTot[k], for velocities at video.Time[k]
'''

#%% 


video.extractGoodFeaturesDisplacementsAccumulateAndInterpolate(display1='quiver',display2='field',displayColor=True,scale=200)
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


video.scaleData(framesPerSecond=20, metersPerPx=0.1, unit=['m', 's'], origin=[0,video.Hvis])

'''
# the scaling function if you want to scale data, i.e., give the fps and the length scale
method is applied, there is no possibility to go back to Unscale, however it is possible to continue to process with scaling


#The Y axis is now oriented upward
#check that with 
'''
video.showXV(video.X, video.V,display='points',displayColor=True)
#or the averaged velocity field

Field = opyf.Render.setField(video.UxTot[0], video.UyTot[0], 'norm')
video.opyfDisp.plotField(Field, vis=video.vis)

#%%

'''
for plotting only the resulting averaged field, usefull if Ntot is longer
'''
video.set_vlim([0,30])
video.set_vecTime(Ntot=10,shift=1,step=2,starting_frame=20)
video.extractGoodFeaturesDisplacementsAccumulateAndInterpolate(display2='field',displayColor=True,scale=200)
video.set_filtersParams(RadiusF=10,maxDevInRadius=2,minNperRadius=3)

#%% 
video.set_trackingFeatures(Ntot=10,step=1, starting_frame=1, track_length=5, detection_interval=10)
'''
# the method {extractTracks} is available to extract tracks on images. The principle is inspired by the openCv sample lktrack.py (https://github.com/opencv/opencv/blob/master/samples/python/lk_track.py)
#The main difference rely on the possibility to store the tracks 
#Note that, if the object is build from a video with videoAnalyzer, a lag is expected since each required images a loaded in the memory for efficiency reasons
It might be a technic to better filter relevant velocities since it is possible to follow these patterns for multiple frames 
'''
opyf.mkdir2('./export_Tracks/')
video.set_filtersParams(wayBackGoodFlag=1,CLAHE=False)
video.extractTracks(display='quiver',displayColor=True, saveImgPath='./export_Tracks/',numberingOutput=True)
# tracks can be saved in csv file
video.writeTracks(outFolder='./export_Tracks')
#when wrtiting it is possible to specify the out folder




