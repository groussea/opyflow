#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 08:24:12 2017

@author: Gauthier ROUSSEAU
"""

# python test file performed on the Case A of the PIV challenge 2014


# 1 #### Initialisation

import vtk  # not used directly in this script but needed
import tqdm
import csv  # not used directly in this script but needed
from custom_cmap import make_cmap
import matplotlib as mpl
import opyf
import json
import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys
import os

folder_main = '.'
os.chdir(folder_main)
# Where is the opyf folder?
sys.path.append('../opyf')

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
mask = cv2.imread('mask.tiff', cv2.IMREAD_ANYDEPTH)

mask = mask/255
plt.imshow(mask)

# %% Load the average Frame
frameav = cv2.imread('averaged_frame.png')

frameav = cv2.cvtColor(frameav, cv2.COLOR_BGR2GRAY)

#%% #####  2   ######### main loop to generate all the detected good points (X,V) with ther respective velocities  ###########
VT_rms = None
incr = 0
plt.close('all')
opyf.Render.opyfFigure()

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
                                                  csvTrack=None,
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
        opyf.Render.opyfPlot(grid_x, grid_y, gridVx, gridVy, X, V,
                             setPlot, vis=frame, namefig='Total2', **render_params)
        plt.savefig(folder_img_output+'/Step'+format(incr,
                                                     '04.0f')+'.png', format='png', dpi=100)

        plt.pause(0.1)
        incr += 1


# %%
setPlot = {'DisplayVis': True,
           'DisplayField': False,
           'QuiverOnFieldColored': False,
           'QuiverOnField': False,
           'DisplayPointsColored': True,
           'DisplayPoints': False,
           'QuiverOnPointsColored': False,
           'QuiverOnPoints': False,
           'DisplayContour': False,
           'ScaleVectors': None,
           # Warning : in Image convention the y0 is on the up left corner
           'extentFrame': [0, Lvis, Hvis, 0],
           'unit': 'px'}
#        For the momentwe didnt interpolate data on grid
opyf.Render.opyfPlot(grid_x, grid_y, gridVx, gridVy, Xdata2,
                     Vdata2, setPlot, vis=frame, namefig='Total2', **render_params)

# %% Filter ouliers
Devx, Npointsx, stDx = opyf.Filters.opyfFindPointsWithinRadiusandDeviation(
    Xdata[:, :], Vdata[:, 0], 5)
Devy, Npointsy, stDy = opyf.Filters.opyfFindPointsWithinRadiusandDeviation(
    Xdata[:, :], Vdata[:, 1], 5)
# %%
Xdata = opyf.Filters.opyfDeletePointCriterion(Xdata, Devx, climmax=0.7)
Vdata = opyf.Filters.opyfDeletePointCriterion(Vdata, Devx, climmax=0.7)
stDx2 = opyf.opyfDeletePointCriterion(stDx, Devx, climmax=0.7)
plt.figure()
nx, binsx, patches = plt.hist(
    stDx, 2000, normed=1, facecolor='red', alpha=0.75, label='DX')


# %%
# Perform statistics on Xdata and Vdata


XT = opyf.Interpolate.npGrid2TargetPoint2D(
    grid_x, grid_y)  # VTK only deal with [X Y] format


# perform analysis directly on Vdata and Xdata
# You can use different interpolation choice
interp_params = dict(Radius=15.,  # it is not necessary to perform unterpolation on a high radius since we have a high number of values
                     Sharpness=20.,
                     kernel='Gaussian',
                     scaleInterp=1.)
VT_mean = opyf.npInterpolateVTK2D(
    Xdata, Vdata, XT, ParametreInterpolatorVTK=interp_params)  # may take a while if X large
gridVx_mean = opyf.Interpolate.npTargetPoints2Grid2D(VT_mean[:, 0], resX, resY)
gridVy_mean = opyf.Interpolate.npTargetPoints2Grid2D(VT_mean[:, 1], resX, resY)


colors = [(11./255, 22./255, 33./255, 1.), (33./255, 66./255, 99./255, 1.),
          (33./255, 66./255, 99./255, 0.), (204./255., 204./255, 0, 1), (0.6, 0, 0, 1)]
position = [0., 0.15/(0.8), 0.2/0.8, 0.3/0.8, 1.]
cmap = make_cmap(colors, position=position)
setPlot = {'DisplayVis': True,
           'DisplayField': True,
           'QuiverOnFieldColored': False,
           'QuiverOnField': False,
           'DisplayPointsColored': False,
           'DisplayPoints': False,
           'QuiverOnPointsColored': False,
           'QuiverOnPoints': False,
           'DisplayContour': False,
           'ScaleVectors': None,
           # Warning : in Image convention the y0 is on the up left corner
           'extentFrame': [0, Lvis, Hvis, 0],
           'unit': 'px'}

render_params = dict(Ptype='horizontal',
                     vlim=[-20., 60.],
                     scale=None)

fig, ax = opyf.Render.opyfPlot(grid_x, grid_y, gridVx_mean, gridVy_mean, Xdata, Vdata,
                               setPlot, vis=frame, namefig='F', cmap=cmap, normalize=True, width=0.0015, **render_params)
opyf.Render.opyfText(fulldict, ax=ax, pos=(600, 900), fontsize=20., alpha=0.4)

# %%


# filter the data to cancel the noise  (ouliers)

Dev, Npoints, stD = opyf.opyfFindPointsWithinRadiusandDeviation(
    XT, VT_full[:, 1], 15)
X1 = opyf.opyfDeletePointCriterion(XT, Dev, climmax=1.)
V1 = opyf.opyfDeletePointCriterion(VT_full, Dev, climmax=1.)
VT_full2 = opyf.npInterpolateVTK2D(
    X1, V1, XT, ParametreInterpolatorVTK=interp_params)

gridVx_mean2 = opyf.Interpolate.npTargetPoints2Grid2D(
    VT_full2[:, 0], resX, resY)
gridVy_mean2 = opyf.Interpolate.npTargetPoints2Grid2D(
    VT_full2[:, 1], resX, resY)
opyf.Render.opyfPlot(grid_x, grid_y, gridVx_mean, gridVy_mean, XT,
                     VT_full, setPlot, vis=frame, namefig='F', cmap=cmap, **render_params)

# Calculate the rms for each cell from the Xdata and Vdata, Trop long....
# il faut limiter à un nombre restrein de vecteur 1000000 de points par exemple
Devx, Npoints, stDx = opyf.Filters.opyfFindPointsWithinRadiusandDeviation(
    Xdata[1:2000000, :], Vdata[1:2000000, 0], 5.)
Devy, Npoints, stDy = opyf.Filters.opyfFindPointsWithinRadiusandDeviation(
    Xdata[1:2000000, :], Vdata[1:2000000, 1], 5.)

# On a un echantillon moyen de 40 points par point, suffisant pour faire des statisitques


# Interpolons les résultats pour la rms
# %%
interp_params = dict(Radius=5.,  # it is not necessary to perform unterpolation on a high radius since we have a high number of values
                     Sharpness=5.,
                     kernel='Gaussian',
                     scaleInterp=1.)
stD_full = opyf.npInterpolateVTK2D(Xdata2, np.transpose(
    [stDx2, stDx2]), XT, ParametreInterpolatorVTK=interp_params)
gridVx_Dev = opyf.Interpolate.npTargetPoints2Grid2D(stD_full[:, 0], resX, resY)
gridVy_Dev = opyf.Interpolate.npTargetPoints2Grid2D(stD_full[:, 1], resX, resY)
colors = [(11./255, 22./255, 33./255, 0.), (33./255, 66./255, 99. /
                                            255, 1.), (204./255., 204./255, 0, 1), (0.6, 0, 0, 1)]
position = [0., 0.15/(0.8), 0.6, 1.]
position = [0., 0.15/(0.8), 0.6, 1.]
cmap = make_cmap(colors, position=position)
cmap.set_under('g')
render_params = dict(Ptype='horizontal',
                     vlim=[0., 10.],
                     scale=1000)
opyf.Render.opyfPlot(grid_x, grid_y, gridVx_Dev, gridVy_Dev, XT, VT_full,
                     setPlot, vis=frame, namefig='F', cmap=cmap, **render_params)

# We obtain a more precised map, bfore, the results were somoothed by the interpolation method giving a fancy plot but hiding more precised
# results

# %%
# build the PIV challenge GRID and interpolate the tracked point on it


gridVx_mean = opyf.Interpolate.npTargetPoints2Grid2D(VT_mean[:, 0], resX, resY)
gridVy_mean = opyf.Interpolate.npTargetPoints2Grid2D(VT_mean[:, 1], resX, resY)
Xvec = grid_x[0, :]
Yvec = grid_y[:, 0]
flag = np.zeros_like(VT_mean[:, 1])
# the function tecplot_WriteRectilinearMesh is inspired by Visit (https://www.visitusers.org) Sources
# it has been used to respect the PIV challenge rules but common csv files could also be generated by
# opyf
variables = (("Vx", VT_mean[:, 0]), ("Vy", VT_mean[:, 1]), ("flag", flag))

opyf.Files.tecplot_WriteRectilinearMesh(
    folder_outputs+'/mean_vec.tec', Xvec, Yvec, variables)
opyf.Files.write_csvTrack2D(folder_outputs+'/mean_vec.csv', XT, VT_mean)

plt.pause(0.1)
ax.invert_yaxis()
opyf.Render.opyfText(fulldict, ax=ax, pos=(600, 900), fontsize=10., alpha=0.4)
fig.set_size_inches([8, 4.5])
ax.set_position([0.13, 0.25, 0.8, 0.7])
plt.savefig(folder_outputs+'/meanFlow2.png', format='png', dpi=100)

# %%
# Another method to obtain de deviation : read the mean file and perform again the loop to calculate at each time step the deviation and finally obtain the rms

out = opyf.tecplot_reader(
    '/media/gauthier/Data_1/TAF/OPyF-Project/Opyf-PIV-Challenge-2014-Test/CaseA/output_with_mask/mean_vec.tec')
VT_mean = out[:, 2:4]
# determine the rms plot
VT_rms = None
incr = 0
for i in tqdm.trange(len(prev)):

    l = listD[select[i]]
    pr = prev[i]
    frame = cv2.imread(folder_src+'/'+l, cv2.IMREAD_ANYDEPTH)
    if pr == False:
        prev_gray = None

#    frame=cv2.imread(folder_src+'/'+l)

    frame = np.array(frame, dtype='float32')
    pxmax = 900.
    pxmin = 350.
    frame = (frame-pxmin)/(pxmax-pxmin)*255
    frame[np.where(frame < 0)] = 0
    frame[np.where(frame > 255)] = 255

    frame = np.array(frame, dtype='uint8')
    frame = opyf.Render.CLAHEbrightness(
        frame, 0, tileGridSize=(20, 20), clipLimit=2)

    # Good feature + flow calculation
    prev_gray, X, V = opyf.Track.opyfFlowGoodFlag(frame, prev_gray, feature_params,
                                                  lk_params, ROI=ROI, vmax=filters_params[
                                                      'vmax'], vmin=filters_params['vmin'],
                                                  csvTrack=None,
                                                  mask=mask, DGF=filters_params['DGF'])

    if len(X) > 0:
        # filters
        Dev, Npoints, stD = opyf.opyfFindPointsWithinRadiusandDeviation(
            X, (V[:, 0]**2+V[:, 1]**2), filters_params['RadiusF'])
        X = opyf.opyfDeletePointCriterion(
            X, Npoints, climmin=filters_params['minNperRadius'])
        V = opyf.opyfDeletePointCriterion(
            V, Npoints, climmin=filters_params['minNperRadius'])
        Dev, Npoints, stD = opyf.opyfFindPointsWithinRadiusandDeviation(
            X, V[:, 1], filters_params['RadiusF'])
        X = opyf.opyfDeletePointCriterion(
            X, Dev, climmax=filters_params['maxDevinRadius'])
        V = opyf.opyfDeletePointCriterion(
            V, Dev, climmax=filters_params['maxDevinRadius'])
        Dev, Npoints, stD = opyf.opyfFindPointsWithinRadiusandDeviation(
            X, V[:, 0], filters_params['RadiusF'])
        X = opyf.opyfDeletePointCriterion(
            X, Dev, climmax=filters_params['maxDevinRadius'])
        V = opyf.opyfDeletePointCriterion(
            V, Dev, climmax=filters_params['maxDevinRadius'])
    if len(X) > 0:
        Xdata = np.append(Xdata, X, axis=0)
        Vdata = np.append(Vdata, V, axis=0)

        resX = 1257
        resY = 592
        grid_y_PIVC, grid_x_PIVC = np.mgrid[24:1208:2, 24:2538:2]

        XT = opyf.Interpolate.npGrid2TargetPoint2D(grid_x_PIVC, grid_y_PIVC)

        VT = opyf.npInterpolateVTK2D(
            X, V, XT, ParametreInterpolatorVTK=interp_params)
        gridVx = opyf.Interpolate.npTargetPoints2Grid2D(VT[:, 0], resX, resY)
        gridVy = opyf.Interpolate.npTargetPoints2Grid2D(VT[:, 1], resX, resY)
        if VT_rms is None:
            VT_rms = np.absolute(VT-VT_mean)
        else:
            VT_rms = VT_rms+np.absolute(VT-VT_mean)
        incr += 1
        VT_rmstemp = VT_rms/incr
        render_params = dict(Ptype='norme',
                             vlim=[0., 20.],
                             scale=1000)
        gridVx_rms = opyf.Interpolate.npTargetPoints2Grid2D(
            VT_rmstemp[:, 0], resX, resY)
        gridVy_rms = opyf.Interpolate.npTargetPoints2Grid2D(
            VT_rmstemp[:, 1], resX, resY)
        setPlot = {'DisplayVis': True,
                   'DisplayField': True,
                   'QuiverOnFieldColored': False,
                   'QuiverOnField': False,
                   'DisplayPointsColored': False,
                   'DisplayPoints': False,
                   'QuiverOnPointsColored': False,
                   'QuiverOnPoints': False,
                   'DisplayContour': False,
                   'ScaleVectors': None,
                   'Text': True,
                   # Warning : in Image convention the y0 is on the up left corner
                   'extentFrame': [0, Lvis, Hvis, 0],
                   'ROI': ROI,
                   'unit': 'px'}
        cmap = opyf.setcmap('norme', alpha=0.5)
        fig, ax = opyf.Render.opyfPlot(grid_x_PIVC, grid_y_PIVC, gridVx_rms, gridVy_rms,
                                       XT, VT_rms, setPlot, vis=frame, namefig='QUiverF', **render_params)

        opyf.Render.opyfText(fulldict, ax=ax, pos=(1200, 1000))
        plt.pause(0.1)
        plt.savefig(folder_img_output3+'/Step'+format(incr,
                                                      '04.0f')+'.png', format='png', dpi=100)


VT_rms = VT_rms/incr
gridVx_mean = opyf.Interpolate.npTargetPoints2Grid2D(VT_mean[:, 0], resX, resY)
gridVy_mean = opyf.Interpolate.npTargetPoints2Grid2D(VT_mean[:, 1], resX, resY)
Xvec = grid_x_PIVC[0, :]
Yvec = grid_y_PIVC[:, 0]
flag = np.zeros_like(VT[:, 1])
variables = (("rms_Vx", VT_rms[:, 0]),
             ("rms_Vy", VT_rms[:, 1]), ("flag", flag))

opyf.Files.tecplot_WriteRectilinearMesh(
    folder_outputs+'/rms_vec.tec', Xvec, Yvec, variables)


# %% plot the rms field with customized color map
setPlot = {'DisplayVis': True,
           'DisplayField': False,
           'QuiverOnFieldColored': False,
           'QuiverOnField': False,
           'DisplayPointsColored': False,
           'DisplayPoints': False,
           'QuiverOnPointsColored': False,
           'QuiverOnPoints': False,
           'DisplayContour': False,
           'ScaleVectors': None,
           'Text': True,
           # Warning : in Image convention the y0 is on the up left corner
           'extentFrame': [0, Lvis, Hvis, 0],
           'ROI': ROI,
           'unit': 'px'}
render_params = dict(Ptype='horizontal',
                     vlim=[0., 10.],
                     scale=1000)


colors = [(11./255, 22./255, 33./255, 0.), (33./255, 66./255, 99. /
                                            255, 1.), (204./255., 204./255, 0, 1), (0.6, 0, 0, 1)]
position = [0., 0.15/(0.8), 0.6, 1.]
cmap = make_cmap(colors, position=position)
cmap.set_under('g')
# cmap.set_over('g')

fig, ax = opyf.Render.opyfPlot(grid_x_PIVC, grid_y_PIVC, gridVx_rms, gridVy_rms,
                               XT, VT_rms, setPlot, vis=frame, cmap=cmap, namefig='QUiverF', **render_params)
resx = grid_x[0, 1]-grid_x[0, 0]
resy = grid_y[1, 0]-grid_y[0, 0]
ROI = [np.min(grid_x)-resx/2, np.min(grid_y)-resy/2, np.max(grid_x) -
       np.min(grid_x)+resx/2, np.max(grid_y)-np.min(grid_y)+resy/2]
#        figp,ax,im=opyfField2(grid_x,grid_y,Field,ax=ax,**infoPlotField)


figp, ax, im = opyf.Render.opyfField(
    gridVx_rms, ax=ax, ROI=ROI, extentr=setPlot['extentFrame'], cmap=cmap, vlim=render_params['vlim'])
figp, cb = opyf.Render.opyfColorBar(
    fig, im, label=' DX rms (in '+setPlot['unit']+'/DeltaT)')
cb.set_alpha(0.8)
cb.draw_all()


plt.pause(0.1)
ax.invert_yaxis()
# opyf.Render.opyfText(fulldict,ax=ax,pos=(600,900),fontsize=10.,alpha=0.8)
fig.set_size_inches([8, 4.5])
ax.set_position([0.13, 0.25, 0.8, 0.7])
plt.savefig(folder_outputs+'/rms.png', format='png', dpi=100)

# %% process the pdf without 0 vectors of the mask area
fig1 = plt.figure()
histDX = VT_mean[np.where(VT_mean[:, 0] != 0.), 0]
histDY = VT_mean[np.where(VT_mean[:, 1] != 0.), 1]
nx, binsx, patches = plt.hist(
    histDX[0], 2000, normed=1, facecolor='red', alpha=0.75, label='DX')
ny, binsy, patches = plt.hist(
    histDY[0], 2000, normed=1, facecolor='green', alpha=0.75, label='DY')
plt.legend()
plt.text(25, 0.25, 'mean DX=' + format(np.mean(histDX), '.2f'))
plt.text(25, 0.22, 'mean DY=' + format(np.mean(histDY), '.2f'))
fig1.set_size_inches([4., 3.])

ax = plt.gca()
ax.set_xlim([-20, 60.])
ax.set_ylim([0., 0.45])
ax.set_position([0.15, 0.2, 0.8, 0.7])
plt.xlabel('DX,DY(in px)')
plt.ylabel('pdf')
plt.savefig(folder_outputs+'/pdf.svg', format='svg', dpi=100)
plt.savefig(folder_outputs+'/pdf.png', format='png', dpi=100)

# %% process to obtain lines as the PIV challenge 2014 publication
plt.close('all')
fig2 = plt.figure()


plt.plot(binsx[0:-1]+(binsx[1]-binsx[0]), nx,
         label='DX', alpha=0.75, linewidth=2.)
plt.plot(binsy[0:-1]+(binsy[1]-binsy[0]), ny,
         label='DY', alpha=0.75, linewidth=2.)
plt.xlabel('DX,DY(in px)')
plt.ylabel('pdf')
plt.text(25, 0.25, 'mean DX=' + format(np.mean(histDX), '.2f'))
plt.text(25, 0.22, 'mean DY=' + format(np.mean(histDY), '.2f'))
ax = plt.gca()
ax.set_xlim([-20, 60.])
ax.set_ylim([0., 0.45])
ax = plt.gca()
fig2.set_size_inches([4., 3.])
ax.set_position([0.15, 0.2, 0.8, 0.7])
plt.legend()


plt.savefig(folder_outputs+'/pdf_line.svg', format='svg', dpi=100)
