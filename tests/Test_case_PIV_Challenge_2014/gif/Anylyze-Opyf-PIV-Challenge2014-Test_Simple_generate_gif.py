#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 08:24:12 2020

@author: Gauthier ROUSSEAU
"""
#%%

# WARNING : to work properly you must run this script from 'Test_case_PIV_Challenge_2014' Folder

# python test file performed on the Case A of the PIV challenge 2014

# This file uses the Anlyze class to reduce the amount of unecessarly scripts

%matplotlib qt5
# 1 #### Initialisation
import sys, os
# if opyf is not installed where is the opyf folder?
sys.path.append('../../')
import opyf

folder_main='.'
# where are the images?
os.chdir(folder_main)
folder_src = folder_main + '/images/'
folder_src ='/media/gauthier/Samsung_T5/images_bis'
# Create a folder to save outputs (csv files, vtk files, hdf5 files, images)
folder_outputs = folder_main+'/outputs'
opyf.mkdir2(folder_outputs)

#For .tif files it is generally usefull to specify the range of Pixels (RangeOfPixels=[px_min,px_max]).
#FOr the PIV challenge case A, pixels range from 450 to 900 in the raw .tif frames.
#%
frames = opyf.frameSequenceAnalyzer(folder_src, imreadOption=2, rangeOfPixels=[450, 900], mask='/media/gauthier/Data-Gauthier/programs/gitHub/opyflow/test/Test_case_PIV_Challenge_2014/mask.tiff')


frames.opyfDisp.ax.set_position([0.12232868757259015,
    0.24062988281250006,
    0.7953426248548199,
    0.6687402343749999])

#frame initial

# frames.Field = np.zeros_like(frames.grid_x)
# frames.opyfDisp.ax.set_position([0.12232868757259015,
# 0.24062988281250006,
# 0.7953426248548199,
# 0.6687402343749999])
# frames.opyfDisp.ax.set_aspect('equal', adjustable='box')
# frames.opyfDisp.fig.set_size_inches(7, 4)
# frames.opyfDisp.ax.set_xlabel('')
# frames.opyfDisp.plotField(frames.Field,vis=frames.vis,vlim=[0,60])


# frames.opyfDisp.ax.set_xlabel('')
# frames.opyfDisp.ax.set_ylim([0, 1230])
# frames.opyfDisp.fig.savefig('/media/gauthier/Data-Gauthier/programs/gitHub/opyflow/test/Test_case_PIV_Challenge_2014/gif/init.png', dpi=142)

#%%
# import numpy as np
frames.set_vlim([0., 60])
# intermediate frames
frames.set_filtersParams(wayBackGoodFlag=1.)
frames.set_goodFeaturesToTrackParams(qualityLevel=0.05,maxCorners=4000)

for s in range(10,100,10):
    frames.set_vecTime(Ntot=10, shift=2, starting_frame=s)
    frames.extractGoodFeaturesDisplacementsAndAccumulate()
    # frames.opyfDisp.plotField(frames.Field,vlim=[0,60])
    frames.opyfDisp.ax.set_position([0.12232868757259015,
    0.24062988281250006,
    0.7953426248548199,
    0.6687402343749999])
    frames.opyfDisp.ax.set_aspect('equal', adjustable='box')
    frames.opyfDisp.fig.set_size_inches(7, 4)
    frames.opyfDisp.ax.set_xlabel('')
    # frames.invertYaxis()
    frames.Vaccu=frames.Vaccu*np.array([1, -1])
    frames.showXV(frames.Xaccu, frames.Vaccu, vis=frames.vis, display='quiver',nvec=5000,vlim=[0,60])
    frames.opyfDisp.ax.set_ylim([0, 1230])
    frames.opyfDisp.fig.savefig('/media/gauthier/Data-Gauthier/programs/gitHub/opyflow/test/Test_case_PIV_Challenge_2014/gif/test'+str(s)+'.png', dpi=142)

#%% generate entire run
frames.set_filtersParams(wayBackGoodFlag=1.,RadiusF=5)
frames.set_opticalFlowParams(maxLevel=3,winSize=(32,16))
frames.set_goodFeaturesToTrackParams(qualityLevel=0.001,maxCorners=6000)
frames.set_vlim([0., 90])
frames.set_vecTime(Ntot=600,shift=2)
frames.extractGoodFeaturesDisplacementsAccumulateAndInterpolate()
frames.set_interpolationParams(Radius=25, Sharpness=4)
frames.interpolateOnGrid(frames.Xaccu,frames.Vaccu)
#%%
# frames.Xaccu, frames.Vaccu=frames.applyFilters(frames.Xaccu, frames.Vaccu)
# frames.interpolateOnGrid(frames.Xaccu, frames.Vaccu)

frames.Field=opyf.Render.setField(frames.Ux, frames.Uy, 'norm')


frames.opyfDisp.ax.set_position([0.12232868757259015,
 0.24062988281250006,
 0.7953426248548199,
 0.6687402343749999])


#%% render
import numpy as np
import matplotlib.pyplot as plt
frames.Field[np.where(frames.Field==0)]=np.nan

frames.Field=frames.Field*frames.gridMask
frames.opyfDisp.plotField(frames.Field, vis=frames.vis, vlim=[0, 60])
frames.opyfDisp.ax.invert_yaxis()
frames.opyfDisp.ax.set_ylim([0, 1230])

frames.opyfDisp.ax.set_position([0.12232868757259015,
 0.24062988281250006,
 0.7953426248548199,
 0.6687402343749999])

frames.opyfDisp.ax.set_aspect('equal', adjustable='box')
frames.opyfDisp.fig.set_size_inches(7, 4)
frames.opyfDisp.ax.set_xlabel('')
plt.show()
frames.opyfDisp.fig.savefig('/media/gauthier/Data-Gauthier/programs/gitHub/opyflow/test/Test_case_PIV_Challenge_2014/gif/frame_1.png', dpi=142)

#%%

# frames.extractGoodFeaturesAndDisplacements()

# #%%


# # =============================================================================
# # Parameters to filter outliers :
# # 'vmin' and 'vmax' determine during the lk_algortihm which vector are directly
# # upressed from the calcultation of velocity.If it is None there ids no effect
# # RadiusF is the radius in which we are looking for neighbours
# # We fix a minimum of N points within RadiusF around each points
# # This strategy supress isolated point that are usally errors
# # maxDevinRadius determine the maximum of deviation of a point compared to his
# # closest N neighbor within RadiusF.
# # 'wayBackGoodFlag' is a condition for the lukas kanade method to considera point if the
# # prediction from A to B is the same than from B to A frame.
# # mask is a binary frame that has the same size than the ROI image.
# #
# # =============================================================================
# filters_params = dict(vmin=None,
#                       vmax=70.,
#                       RadiusF=10.,
#                       minNperRadius=2.,
#                       maxDevinRadius=3.,
#                       wayBackGoodFlag=1.,
#                       mask=None)

# # To interpolate datas we use vtk librarie which provide in its last versions efficient
# # Interpolation calculation
# #
# interp_params = dict(Radius=15.,
#                      Sharpness=2.,
#                      kernel='Gaussian',
#                      scaleInterp=1.)


# # Build the ouput grid fixed by the challenge rules

# resX = 1257
# resY = 592
# grid_y, grid_x = np.mgrid[24:1208:2, 24:2538:2]
# gridVx, gridVy = np.zeros_like(grid_x), np.zeros_like(grid_x)

# # (it is possible to build another grid and do interpolation on a specific ROI if needed)
# # We define here the resolution for the interpolation
# # the scale is usally higher (a resolution a bit lower than the image) but its
# # also possible to increase this resolution for special needs


# # %%

# #compute the Average frame to determine the mask and substract to cancel mean noise  ##################
# # take a while to compute so you can directly load the saved averaged image
# # frameav=None
# #
# # for i in range(len(listD)):
# # incr=0
# # nframe=1200
# # for i in range(nframe):
# #    incr+=1
# #    l=listD[select[i]]
# #    pr=prev[i]
# #    frame=cv2.imread(folder_src+'/'+l,cv2.IMREAD_ANYDEPTH)
# #    if pr==False:
# #        prev_gray=None
# #
# # frame=cv2.imread(folder_src+'/'+l)
# #
# #    frame=np.array(frame,dtype='float32')
# #    pxmax=900.
# #    pxmin=450.
# #    frame=(frame-pxmin)/(pxmax-pxmin)
# #    frame[np.where(frame<0.)]=0.
# #    frame[np.where(frame>1.)]=1.
# #
# #    if frameav is None:
# #        frameav=frame
# #    else:
# #        frameav=frameav+frame
# #
# #
# # frameav=frameav/incr
# #
# # plt.imshow(frameav)
# # cv2.imwrite(folder_outputs+'/frame.png',frameav*255)


# # %% After a treatment with image J (or other image treatment software) we are able to take the binary image mask.tiff
# # The file is available in the local repository
# mask = cv2.imread(folder_main+'/mask.tiff', cv2.IMREAD_ANYDEPTH)

# mask = mask/255


# # %% Load the average Frame
# frameav = cv2.imread('averaged_frame.png')

# frameav = cv2.cvtColor(frameav, cv2.COLOR_BGR2GRAY)

# #%% #####  2   ######### main loop to generate all the detected good points (X,V) with ther respective velocities  ###########
# VT_rms = None
# incr = 0
# plt.close('all')
# # opyf.Render.opyfFigure()

# for i in tqdm.trange(len(prev)):

#     l = listD[select[i]]
#     pr = prev[i]
#     # special for tiff images
#     frame = cv2.imread(folder_src+'/'+l, cv2.IMREAD_ANYDEPTH)
#     if pr == False:
#         prev_gray = None

# #    frame=cv2.imread(folder_src+'/'+l)
# # series of treatment to obtain a scaled image from the raw tiff images provided by PIV challenge 2014
#     frame = np.array(frame, dtype='float32')
#     pxmax = 900.
#     pxmin = 450.

#     frame = ((frame-pxmin)/(pxmax-pxmin))*255-frameav

#     frame[np.where(frame < 0)] = 0
#     frame[np.where(frame > 255)] = 255
# #    plt.clf()
# #    plt.imshow(frame)
# #    plt.pause(0.1)

#     # Opencv only deal with uint8 images
#     frame = np.array(frame, dtype='uint8')
# # The results are highly dependant on how we enhance the images
# # The function CLAHE help to enhance and obtain good balance between the two images
#     frame = opyf.Render.CLAHEbrightness(
#         frame, 0, tileGridSize=(20, 20), clipLimit=2)
# # plt.imshow(frame)
# # folder_outputs+'/'+format(incr,'04.0f')+'.csv'

#     # Good feature + flow calculatition
#     prev_gray, X, V = opyf.Track.opyfFlowGoodFlag(frame, prev_gray, feature_params,
#                                                   lk_params, ROI=ROI, vmax=filters_params[
#                                                       'vmax'], vmin=filters_params['vmin'],
#                                                   mask=mask, wayBackGoodFlag=filters_params['wayBackGoodFlag'])


# #    if len(X)>0:
# #        #filters ###### important since we have a low quality level for the Good Feature to track and a high Distance Good Flag
# #        Dev,Npoints,stD=opyf.opyfFindPointsWithinRadiusandDeviation(X,(V[:,0]**2+V[:,1]**2),filters_params['RadiusF'])
# #        X=opyf.opyfDeletePointCriterion(X,Npoints,climmin=filters_params['minNperRadius'])
# #        V=opyf.opyfDeletePointCriterion(V,Npoints,climmin=filters_params['minNperRadius'])
# #        Dev,Npoints,stD=opyf.opyfFindPointsWithinRadiusandDeviation(X,V[:,1],filters_params['RadiusF'])
# #        X=opyf.opyfDeletePointCriterion(X,Dev,climmax=filters_params['maxDevinRadius'])
# #        V=opyf.opyfDeletePointCriterion(V,Dev,climmax=filters_params['maxDevinRadius'])
# #        Dev,Npoints,stD=opyf.opyfFindPointsWithinRadiusandDeviation(X,V[:,0],filters_params['RadiusF'])
# #        X=opyf.opyfDeletePointCriterion(X,Dev,climmax=filters_params['maxDevinRadius'])
# #        V=opyf.opyfDeletePointCriterion(V,Dev,climmax=filters_params['maxDevinRadius'])
#     if len(X) > 0:
#         Xdata = np.append(Xdata, X, axis=0)
#         Vdata = np.append(Vdata, V, axis=0)
#         # Many rendering are possible here are 3 important parameters.
#         # the lgortihm can plot 'horizontal', 'vertical' or norme field values
#         #
#         # for the vectos
#         render_params = dict(Ptype='norm',
#                              vlim=[0, 70.],
#                              scale=1000)

#         # set Plot is a dictionnary that determine which type of plot we want
#         # it is possible to superimpose informations (vectors + color map for instance)
#         setPlot = {'DisplayVis': True,
#                    'DisplayField': False,
#                    'QuiverOnFieldColored': False,
#                    'QuiverOnField': False,
#                    'DisplayPointsColored': False,
#                    'DisplayPoints': False,
#                    'QuiverOnPointsColored': True,
#                    'QuiverOnPoints': False,
#                    'DisplayContour': False,
#                    'ScaleVectors': None,
#                    # Warning : in Image convention the y0 is on the up left corner
#                    'extentFrame': [0, Lvis, Hvis, 0],
#                    'unit': 'px'}
# #        For the momentwe didnt interpolate data on grid

#         opyf.Render.opyfPlot(grid_x, grid_y, gridVx, gridVy, X,
#                              V, setPlot, vis=frame, namefig='Vel', **render_params)
#         plt.savefig(folder_img_output+'/Test_Step' +
#                     format(incr, '04.0f')+'.png', format='png', dpi=100)

#         plt.pause(0.1)
#         incr += 1

#     # %% Save 3D velocities
