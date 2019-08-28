#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 08:24:12 2017

@author: Gauthier ROUSSEAU
"""

# WARNING : to work properly you must run this script from 'Test_case_PIV_Challenge_2014' Folder

# python test file performed on the Case A of the PIV challenge 2014


# 1 #### Initialisation

import tqdm
import opyf
import json
import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys
import os

folder_main = './'
os.chdir(folder_main)
# if it is not installed where is the opyf folder?
# sys.path.append('/media/gauthier/Data_1/TAF/OPyF-Project/github/opyFlow')

plt.close('all')


# where are the images?
folder_src = folder_main + '/images'

# Create a folder to save outputs (csv files, vtk files, images)
folder_outputs = folder_main+'/outputs'
opyf.mkdir2(folder_outputs)


# create an image output folder
folder_img_output = folder_outputs+'/images'

opyf.mkdir2(folder_img_output)

listD = os.listdir(folder_src)
listD = np.sort(listD)


# Calculate the main proprieties of the frames
# frameini=cv2.imread(folder_src+'/'+listD[0])
frameini = cv2.imread(folder_src+'/'+listD[0], cv2.IMREAD_ANYDEPTH)
Hvis, Lvis = frameini.shape
ROI = [0, 0, frameini.shape[1], frameini.shape[0]]

# It is possible to specify the ROI to accelerate the treatment
# ROI=[0,602,2799,1252]


# How are structured the images 'ABAB' or 'ABCD' (with the same time interval).
# It is also possible to introduce a shift if we want to
seqIm_params = dict(seqType='ABAB',
                    shift=2)

# this fuctntion will produce 2 vectors
# select will select the index of the selected frame in listD
# prev is a boolean vector which indicate if there is an image to consider before or not
select, prev = opyf.Files.initializeSeqVec(seqIm_params, listD)

# Parameters for the Good Feature to Track algorithm (Shi-Tomasi Corner Detector)
# the more we consider corners, the more we are able to reproduce the velocity
# be careful that whith a too low quality level for vectors the results are poor
# normal filters are needed to exclude bad vectors
# he nin distance is the minimum distance between vectors

feature_params = dict(maxCorners=70000,
                      qualityLevel=0.09,
                      minDistance=4,
                      blockSize=16)

# Parameters for the flow calculation using Lukas Kanade method
# WinSize caracterise the size of the window in which we search movement
# Warning : the algorithm is pyramidal. For the first step the
lk_params = dict(winSize=(32, 6),
                 maxLevel=4,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.03))


# =============================================================================
# Parameters to filter outliers :
# 'vmin' and 'vmax' determine during the lk_algortihm which vector are directly
# upressed from the calcultation of velocity.If it is None there ids no effect
# RadiusF is the radius in which we are looking for neighbours
# We fix a minimum of N points within RadiusF around each points
# This strategy supress isolated point that are usally errors
# maxDevinRadius determine the maximum of deviation of a point compared to his
# closest N neighbor within RadiusF.
# 'DGF' is a condition for the lukas kanade method to considera point if the
# prediction from A to B is the same than from B to A frame.
# mask is a binary frame that has the same size than the ROI image.
#
# =============================================================================
filters_params = dict(vmin=None,
                      vmax=70.,
                      RadiusF=10.,
                      minNperRadius=2.,
                      maxDevinRadius=3.,
                      DGF=1.,
                      mask=None)

# To interpolate datas we use vtk librarie which provide in its last versions efficient
# Interpolation calculation
#
interp_params = dict(Radius=15.,
                     Sharpness=2.,
                     kernel='Gaussian',
                     scaleInterp=1.)


# Build the ouput grid fixed by the challenge rules

resX = 1257
resY = 592
grid_y, grid_x = np.mgrid[24:1208:2, 24:2538:2]
gridVx, gridVy = np.zeros_like(grid_x), np.zeros_like(grid_x)

# (it is possible to build another grid and do interpolation on a specific ROI if needed)
# We define here the resolution for the interpolation
# the scale is usally higher (a resolution a bit lower than the image) but its
# also possible to increase this resolution for special needs

# framecut=frameini[ROI[1]:(ROI[3]+ROI[1]),ROI[0]:(ROI[2]+ROI[0])]
# scaleinterp=int(interp_params['scaleInterp'])
# [Ly,Cx]=framecut.shape
# resX=Cx/scaleinterp
# resY=Ly/scaleinterp
#
#grid_y, grid_x = np.mgrid[0:resY, 0:resX]
# grid_x=(grid_x+0.5)*np.float(ROI[2])/resX+ROI[0]
# grid_y=(grid_y+0.5)*np.float(ROI[3])/resY+ROI[1]
#
# gridVx,gridVy=np.zeros_like(grid_x),np.zeros_like(grid_x)


# Save the parameters into a json dictionnary file

fulldict = {'lk_params': lk_params, 'feature_params': feature_params,
            'filters_params': filters_params,
            'interp_params': interp_params}

out_file_name = folder_img_output+'/parametres.json'

out_file = open(out_file_name, "w")
# (the 'indent=4' is optional, but makes it more readable)
json.dump(fulldict, out_file, indent=4)
out_file.close()


#fig1=plt.figure('track',figsize=(Lfig, Hfig))

prev_gray = None
tracks = []
Xdata = np.empty([0, 2])
Vdata = np.empty([0, 2])
Xnul = []
Vnul = []


# %%

#compute the Average frame to determine the mask and substract to cancel mean noise  ##################
# take a while to compute so you can directly load the saved averaged image
# frameav=None
#
# for i in range(len(listD)):
# incr=0
# nframe=1200
# for i in range(nframe):
#    incr+=1
#    l=listD[select[i]]
#    pr=prev[i]
#    frame=cv2.imread(folder_src+'/'+l,cv2.IMREAD_ANYDEPTH)
#    if pr==False:
#        prev_gray=None
#
# frame=cv2.imread(folder_src+'/'+l)
#
#    frame=np.array(frame,dtype='float32')
#    pxmax=900.
#    pxmin=450.
#    frame=(frame-pxmin)/(pxmax-pxmin)
#    frame[np.where(frame<0.)]=0.
#    frame[np.where(frame>1.)]=1.
#
#    if frameav is None:
#        frameav=frame
#    else:
#        frameav=frameav+frame
#
#
# frameav=frameav/incr
#
# plt.imshow(frameav)
# cv2.imwrite(folder_outputs+'/frame.png',frameav*255)


# %% After a treatment with image J (or other image treatment software) we are able to take the binary image mask.tiff
# The file is available in the local repository
mask = cv2.imread(folder_main+'/mask.tiff', cv2.IMREAD_ANYDEPTH)

mask = mask/255


# %% Load the average Frame
frameav = cv2.imread('averaged_frame.png')

frameav = cv2.cvtColor(frameav, cv2.COLOR_BGR2GRAY)

#%% #####  2   ######### main loop to generate all the detected good points (X,V) with ther respective velocities  ###########
VT_rms = None
incr = 0
plt.close('all')
# opyf.Render.opyfFigure()

for i in tqdm.trange(len(prev)):

    l = listD[select[i]]
    pr = prev[i]
    # special for tiff images
    frame = cv2.imread(folder_src+'/'+l, cv2.IMREAD_ANYDEPTH)
    if pr == False:
        prev_gray = None

#    frame=cv2.imread(folder_src+'/'+l)
# series of treatment to obtain a scaled image from the raw tiff images provided by PIV challenge 2014
    frame = np.array(frame, dtype='float32')
    pxmax = 900.
    pxmin = 450.

    frame = ((frame-pxmin)/(pxmax-pxmin))*255-frameav

    frame[np.where(frame < 0)] = 0
    frame[np.where(frame > 255)] = 255
#    plt.clf()
#    plt.imshow(frame)
#    plt.pause(0.1)

    # Opencv only deal with uint8 images
    frame = np.array(frame, dtype='uint8')
# The results are highly dependant on how we enhance the images
# The function CLAHE help to enhance and obtain good balance between the two images
    frame = opyf.Render.CLAHEbrightness(
        frame, 0, tileGridSize=(20, 20), clipLimit=2)
# plt.imshow(frame)
# folder_outputs+'/'+format(incr,'04.0f')+'.csv'

    # Good feature + flow calculatition
    prev_gray, X, V = opyf.Track.opyfFlowGoodFlag(frame, prev_gray, feature_params,
                                                  lk_params, ROI=ROI, vmax=filters_params[
                                                      'vmax'], vmin=filters_params['vmin'],
                                                  mask=mask, DGF=filters_params['DGF'])


#    if len(X)>0:
#        #filters ###### important since we have a low quality level for the Good Feature to track and a high Distance Good Flag
#        Dev,Npoints,stD=opyf.opyfFindPointsWithinRadiusandDeviation(X,(V[:,0]**2+V[:,1]**2),filters_params['RadiusF'])
#        X=opyf.opyfDeletePointCriterion(X,Npoints,climmin=filters_params['minNperRadius'])
#        V=opyf.opyfDeletePointCriterion(V,Npoints,climmin=filters_params['minNperRadius'])
#        Dev,Npoints,stD=opyf.opyfFindPointsWithinRadiusandDeviation(X,V[:,1],filters_params['RadiusF'])
#        X=opyf.opyfDeletePointCriterion(X,Dev,climmax=filters_params['maxDevinRadius'])
#        V=opyf.opyfDeletePointCriterion(V,Dev,climmax=filters_params['maxDevinRadius'])
#        Dev,Npoints,stD=opyf.opyfFindPointsWithinRadiusandDeviation(X,V[:,0],filters_params['RadiusF'])
#        X=opyf.opyfDeletePointCriterion(X,Dev,climmax=filters_params['maxDevinRadius'])
#        V=opyf.opyfDeletePointCriterion(V,Dev,climmax=filters_params['maxDevinRadius'])
    if len(X) > 0:
        Xdata = np.append(Xdata, X, axis=0)
        Vdata = np.append(Vdata, V, axis=0)
        # Many rendering are possible here are 3 important parameters.
        # the lgortihm can plot 'horizontal', 'vertical' or norme field values
        #
        # for the vectos
        render_params = dict(Ptype='norme',
                             vlim=[0, 70.],
                             scale=1000)

        # set Plot is a dictionnary that determine which type of plot we want
        # it is possible to superimpose informations (vectors + color map for instance)
        setPlot = {'DisplayVis': True,
                   'DisplayField': False,
                   'QuiverOnFieldColored': False,
                   'QuiverOnField': False,
                   'DisplayPointsColored': False,
                   'DisplayPoints': False,
                   'QuiverOnPointsColored': True,
                   'QuiverOnPoints': False,
                   'DisplayContour': False,
                   'ScaleVectors': None,
                   # Warning : in Image convention the y0 is on the up left corner
                   'extentFrame': [0, Lvis, Hvis, 0],
                   'unit': 'px'}
#        For the momentwe didnt interpolate data on grid

        opyf.Render.opyfPlot(grid_x, grid_y, gridVx, gridVy, X,
                             V, setPlot, vis=frame, namefig='Vel', **render_params)
        plt.savefig(folder_img_output+'/Test_Step' +
                    format(incr, '04.0f')+'.png', format='png', dpi=100)

        plt.pause(0.1)
        incr += 1

    # %% Save 3D velocities
