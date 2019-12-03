#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 08:59:41 2019

@author: Gauthier ROUSSEAU
"""

import os
import sys
import numpy as np
import cv2
from opyf import MeshesAndTime, Track, Render, Files, Tools, Interpolate, Filters
import matplotlib.pyplot as plt
import time


class Analyzer():
    def __init__(self, imageROI=None, **args):

        print('Dimensions :\n \t', 'Width :',
              self.frameInit.shape[1], 'Height :', self.frameInit.shape[0])

        self.scaled = False
        self.scale= 1
        if imageROI is None:
            self.ROI = [0, 0, self.frameInit.shape[1], self.frameInit.shape[0]]
        else:
            self.ROI = imageROI
        print('Regio Of Interest :\n \t', self.ROI)
        self.cropFrameInit = self.frameInit[self.ROI[1]:(
            self.ROI[3]+self.ROI[1]), self.ROI[0]:(self.ROI[2]+self.ROI[0])]
        self.Hvis, self.Lvis = self.cropFrameInit.shape
        self.mask = np.ones(self.cropFrameInit.shape)
        self.frameAv = np.zeros(self.cropFrameInit.shape)
        self.set_goodFeaturesToTrackParams()
        self.set_opticalFlowParams()
        self.set_filtersParams()
        self.set_interpolationParams()
        # self.set_tracksParams()
        self.vlimPx = args.get('vlim', [-np.inf, np.inf])
        self.prevTracks = None


# TODO extract samples + function        self.samplesMat = []
#        self.Xsamples = []

        self.unit = args.get('unit', ['px', 'deltaT'])

        self.set_imageParams(**args)
        self.paramPlot = {'ScaleVectors': args.get('ScaleVectors', 0.1),
                          'vecX': [],
                          'vecY': [],
                          'extentFrame': args.get('extentFrame', [0, self.Lvis, self.Hvis, 0]),
                          'unit': args.get('unit', ['px', 'deltaT']),
                          'Hfig': args.get('Hfig', 8),
                          'grid': args.get('grid', True),
                          'vlim': args.get('vlim', [0, 40])}
        self.reset()
        self.set_gridToInterpolateOn(0, self.Lvis, 4, 0, self.Hvis, 4)
        self.gridMask = np.ones((self.Hgrid, self.Lgrid))

        self.set_vecTime()
        print('number of frames : ' + str(self.number_of_frames))
        self.fieldResults=None
        # plt.ion()

    def set_goodFeaturesToTrackParams(self, maxCorners=40000, qualityLevel=0.005,
                                      minDistance=5, blockSize=16):

        self.feature_params = dict(maxCorners=maxCorners,
                                   qualityLevel=qualityLevel,
                                   minDistance=minDistance,
                                   blockSize=blockSize)
        print('')
        print('Good Features To Track Parameters:')
        for x in self.feature_params:
            print('\t- ', x, ':', self.feature_params[x])

    def set_opticalFlowParams(self, winSize=(16, 16), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.03)):

        self.lk_params = dict(winSize=winSize,
                              maxLevel=maxLevel,
                              criteria=criteria)
        print('')
        print('Optical Flow Parameters:')
        for x in self.lk_params:
            print('\t- ', x, ':', self.lk_params[x])

    def set_filtersParams(self, RadiusF=30, minNperRadius=0, maxDevInRadius=np.inf, wayBackGoodFlag=np.inf,CLAHE=False):
        self.filters_params = dict(RadiusF=RadiusF,
                                   minNperRadius=minNperRadius,
                                   maxDevInRadius=maxDevInRadius,
                                   wayBackGoodFlag=wayBackGoodFlag,
                                   CLAHE=CLAHE)

        print('')
        print('Filters Params:')
        for x in self.filters_params:
            print('\t- ', x, ':', self.filters_params[x])

    def set_interpolationParams(self, Radius=30, Sharpness=8, kernel='Gaussian'):
        self.interp_params = dict(Radius=Radius,  # it is not necessary to perform unterpolation on a high radius since we have a high number of values
                                  Sharpness=Sharpness,
                                  kernel=kernel)
        print('')
        print('Interpolation Parameters:')
        for x in self.interp_params:
            print('\t- ', x, ':', self.interp_params[x])

    def set_vlim(self, vlim):
        if self.scaled == False:
            self.vlimPx = vlim
        else:
            self.vlimPx = [
                v/(self.scale*self.fps/self.paramVecTime['step']) for v in vlim]

        self.paramPlot['vlim'] = vlim
        plt.close('opyfPlot')
        self.opyfDisp = Render.opyfDisplayer(**self.paramPlot, num='opyfPlot')
        print('Velocity limits: ', vlim[0])
        print('\t minimum norm velocity: ', vlim[0])
        print('\t maximum norm velocity: ', vlim[1])

    def set_gridToInterpolateOn(self, pixLeft=0, pixRight=0, stepHor=2, pixUp=0, pixDown=0, stepVert=2, ROI=None, stepGrid=None):
        if pixRight == 0:
            pixRight = self.Lvis
        if pixDown == 0:
            pixDown = self.Hvis
        if stepGrid is not None:
            stepVert, stepHor = stepGrid, stepGrid
        if ROI is not None:
            self.ROImeasure = ROI
            pixLeft, pixRight, pixUp, pixDown = ROI[0], ROI[0] + \
                ROI[2], ROI[1],  ROI[1]+ROI[3]
        else:
            self.ROImeasure = [pixLeft, pixUp, pixRight-pixLeft, pixDown-pixUp]
        self.grid_y, self.grid_x, self.gridVx, self.gridVy,  self.Hgrid, self.Lgrid = MeshesAndTime.set_gridToInterpolateOn(
            pixLeft, pixRight, stepHor, pixUp, pixDown, stepVert)
        if self.scaled == True:
            self.grid_y, self.grid_x = - \
                (self.grid_y-self.origin[1]) * \
                self.scale, (self.grid_x-self.origin[0])*self.scale
        self.vecX = self.grid_x[0, :]
        self.vecY = self.grid_y[:, 0]
        self.XT = Interpolate.npGrid2TargetPoint2D(self.grid_x, self.grid_y)

        self.paramPlot['vecX'] = self.vecX
        self.paramPlot['vecY'] = self.vecY
        self.opyfDisp = Render.opyfDisplayer(**self.paramPlot, num='opyfPlot')
        self.Ux, self.Uy = np.zeros((self.Hgrid, self.Lgrid)), np.zeros(
            (self.Hgrid, self.Lgrid))
        self.gridMask = np.ones((self.Hgrid, self.Lgrid))

    def set_imageParams(self, **args):
        self.rangeOfPixels = args.get('rangeOfPixels', [0, 255])
        self.imreadOption = args.get('imreadOption', 1)
        if self.imreadOption == 'ANYDEPTH':
            self.imreadOption = 2

    def set_vecTime(self, starting_frame=0, step=1, shift=1, Ntot=1):
        self.paramVecTime = {'starting_frame': starting_frame,
                             'step': step,
                             'shift': shift,
                             'Ntot':  Ntot}

        self.vec, self.prev = MeshesAndTime.set_vecTime(starting_frame=self.paramVecTime['starting_frame'],
                                                        step=self.paramVecTime['step'],
                                                        shift=self.paramVecTime['shift'],
                                                        Ntot=self.paramVecTime['Ntot'])
        if self.vec[-1] > self.number_of_frames:
            print('----- Error ----')
            print('Your processing plan is not compatible with the frame set')
            print(
                'Consider that [starting_frame+step+(Ntot-1)*shift]  must be smaller than the number of frames')
            sys.exit()
        print('\n[The following image processing plan has been set]')
        if self.processingMode == 'image sequence':
            for pr, i in zip(self.prev, self.vec):
                if pr == False:
                    print(
                        '--> detect Good Features to Track on image [' + self.listD[i]+']')
                    file_prev = self.listD[i]
                else:
                    print(
                        '--> diplacements measurement between images [' + file_prev + '] and [' + self.listD[i] + ']')

        if self.processingMode == 'video':
            for pr, i in zip(self.prev, self.vec):
                if pr == False:
                    print(
                        '--> detect Good Features to Track on frame [' + str(i)+']')
                    file_prev = str(i)
                else:
                    print(
                        '--> diplacements measurement between frame [' + file_prev + '] and [' + str(i) + ']')
        self.Time = (self.vec[0:-1:2]+self.vec[1::2])/2
        # initilize self.vis with the first frame of the set

        if self.processingMode == 'video':
            self.dictFrames = {}
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.sortedVec = np.sort(np.unique(self.vec))
            k = 0
            while k < len(self.sortedVec):
                indF = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                ret, vis = self.cap.read()
                if indF == self.sortedVec[k]:
                    self.dictFrames[str(self.sortedVec[k])] = vis
                    k += 1
        self.readFrame(self.vec[0])

    def set_trackingFeatures(self, starting_frame=0, step=1, Ntot=10, track_length=10, detection_interval=10):

        self.tracks_params = {'track_len': track_length,
                              'detect_interval': detection_interval}

        self.paramVecTimeTracks = {'starting_frame': starting_frame,
                                   'step': step,
                                   'Ntot':  Ntot}

        self.vecTracks, self.prevTracks = MeshesAndTime.set_vecTimeTracks(
            starting_frame=starting_frame, step=step,  Ntot=Ntot)

        print(
            '\nTracking processing \nTracking operates only in succesives images sperated with [step] frames')
        if self.vec[-1] > self.number_of_frames:
            print('----- Error ----')
            print('Your tracking plan is not compatible with the frame set')
            print(
                'Consider that for tracking [starting_frame+step*(Ntot)]  must be smaller than the number of frames')
            sys.exit()

        print('Starting frame : [' + str(starting_frame) + ']')

        print('--> Tracking operate from frame [' + str(
            self.vecTracks[1]) + '] to [' + str(self.vecTracks[-1]) + ']')

        print('On the first frame, only Good Features to Track are detected')

        print('Maximum tracking length are ['+str(track_length)+']')
        print(
            'Good features detection are reloaded every ['+str(detection_interval)+'] frames')
        print(
            'After[extractTracks] method, tracks are stored in [tracks], \nFor saving them run [writeTracks]')

        self.set_filtersParams(wayBackGoodFlag=1)

        if self.processingMode == 'video':
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.sortedVec = np.sort(np.unique(self.vecTracks))
            k = 0
            while k < len(self.sortedVec):
                indF = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                ret, vis = self.cap.read()
                if indF == self.sortedVec[k]:
                    self.dictFrames[str(self.sortedVec[k])] = vis
                    k += 1
        self.readFrame(self.vec[0])

    def initializeAveragedFrameFromFile(self, file, imreadOption=1):
        frameav = cv2.imread(file, imreadOption)
        frameav = Tools.convertToGrayScale(frameav)
        if frameav is None:
            print('Invalid path for the averaged frame')
        elif frameav.shape != (self.Hvis, self.Lvis):
            print('Invalid dimensions for the averaged frame')
        else:
            self.frameAv = frameav

# TODO and test
#    def initializeAveragedFrameFromSequence(self,vec):
#        frameav=None
#        incr=0
#        for i in vec:
#            incr+=1
#            l=self.listD[i]
#            if self.processingMode=='image sequence':
#                frame=cv2.imread(self.folder_src+'/'+l,self.imreadOption)
#                frame=Tools.convertToGrayScale(frame)
#        #    frame=cv2.imread(folder_src+'/'+l)
#            frame=(frame-self.rangeOfPixels[0])/(self.rangeOfPixels[1]-self.rangeOfPixels[0])*255
#            frame[np.where(frame<0)]=0
#            frame[np.where(frame>255)]=255
#
#            if frameav is None:
#                frameav=frame
#            else:
#                frameav=frameav+frame
#            frameav=frameav/incr
#            self.frameAv=frameav
    def readFrame(self, i):
        if self.processingMode == 'image sequence':
            l = self.listD[i]
            self.vis = cv2.imread(self.folder_src+'/'+l,
                                  self.imreadOption)  # 2 for tiff images
        elif self.processingMode == 'video':
            self.vis = self.dictFrames[str(i)]

        self.vis = self.vis[self.ROI[1]:(
            self.ROI[3]+self.ROI[1]), self.ROI[0]:(self.ROI[2]+self.ROI[0])]

    # TODO  def stepDeepFlow(self, pr, i):
        # self.readFrame(i)
        # from deepmatching import deepmatching
        # from deepflow2 import deepflow2
        # if pr == False:
        #     self.prev_col = None
        # else:
        #     self.matches = deepmatching(self.prev_col, self.vis)
        #     self.flow = deepflow2(self.prev_col, self.vis,
        #                           self.matches, '-sintel')
        # self.prev_col = np.copy(self.vis)

    # def extractDeepFlow(self, display=False, saveImgPath=None, Type='norme', imgFormat='png', **args):
    #     self.reset()
    #     for pr, i in zip(self.prev, self.vec):
    #         self.stepDeepFlow(pr, i)
    #         if pr == True:
    #             self.Ux = self.flow[:, :, 0]
    #             self.Uy = self.flow[:, :, 1]
    #             self.UxTot.append(self.flow[:, :, 0])
    #             self.UyTot.append(self.flow[:, :, 1])
    #             Field = Render.setField(self.Ux, self.Uy, Type)

    #             if display == 'field':
    #                 self.opyfDisp.plotField(Field, vis=self.vis, **args)
    #                 if saveImgPath is not None:
    #                     self.opyfDisp.fig.savefig(saveImgPath+'/'+display+'_'+format(
    #                         i, '04.0f')+'_to_'+format(i+self.paramVecTime['step'], '04.0f')+'.'+imgFormat)
    #                 plt.show()
    #                 plt.pause(0.02)

    def stepTracks(self, pr, i):
        if pr == False:
            self.prev_gray = None

        self.readFrame(i)
        self.substractAveragedFrame()
        self.tracks, self.vtracks, self.prev_gray, self.X, self.V = Track.opyfTrack(self.tracks, self.vtracks, self.gray, self.prev_gray,
                                                                                    self.incr, self.feature_params,
                                                                                    self.lk_params, self.tracks_params,
                                                                                    ROI=self.ROImeasure,
                                                                                    vmin=self.vlimPx[0],
                                                                                    vmax=self.vlimPx[1],
                                                                                    mask=self.mask,
                                                                                    wayBackGoodFlag=self.filters_params['wayBackGoodFlag'])

        self.scaleAndLogTracks(i)

    def extractTracks(self, display=False, saveImgPath=None, numberingOutput=False,imgFormat='png', **args):
        self.reset()
        self.tracks = []
        self.vtracks = []
        if self.prevTracks is None:
            print(
                '\n \nWARNING : To run the extractTracks() method, it is mandatory to define the tracking plan through the method [set_trackingFeatures()]\n\n')
            sys.exit()
        k=0
        for pr, i in zip(self.prevTracks, self.vecTracks):
            self.stepTracks(pr, i)
            if pr == True:
                self.Xdata.append(self.X)
                self.Vdata.append(self.V)

                cv2.polylines(self.vis,
                              [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
                self.showXV(self.X, self.V, vis=self.vis,
                            display=display, **args)
                if saveImgPath is not None:
                    if numberingOutput==True:
                        self.opyfDisp.fig.savefig(saveImgPath+'/'+display+'_'+format(k, '04.0f')+'.'+imgFormat) 
                        k+=1
                    else: 
                        self.opyfDisp.fig.savefig(saveImgPath+'/'+display+'_'+format(
                        i, '04.0f')+'_to_'+format(i+self.paramVecTime['step'], '04.0f')+'.'+imgFormat)
                        
                if display is not False:
                    self.opyfDisp.fig.show()
                    plt.pause(0.02)

    def substractAveragedFrame(self):
        self.gray = Tools.convertToGrayScale(self.vis)
        if self.rangeOfPixels != [0, 255] or self.frameAv is not None:
            self.gray = np.array(self.gray, dtype='float32')
            self.gray = (
                (self.gray-self.rangeOfPixels[0])/(self.rangeOfPixels[1]-self.rangeOfPixels[0]))*255-self.frameAv
            self.gray[np.where(self.gray < 0)] = 0
            self.gray[np.where(self.gray > 255)] = 255
            self.gray = np.array(self.gray, dtype='uint8')

    def stepGoodFeaturesToTrackandOpticalFlow(self, pr, i):
        if pr == False:
            self.prev_gray = None
        self.readFrame(i)
        self.substractAveragedFrame()


        if self.filters_params['CLAHE']==True:
             self.gray= Render.CLAHEbrightness(
             self.gray, 0, tileGridSize=(20, 20), clipLimit=2)
#             self.vis=Render.CLAHEbrightness(self.vis,0,tileGridSize=(20,20),clipLimit=2)

        self.prev_gray, self.X, self.V = Track.opyfFlowGoodFlag(self.gray,
                                                                self.prev_gray,
                                                                self.feature_params,
                                                                self.lk_params,
                                                                ROI=self.ROImeasure,
                                                                vmin=self.vlimPx[0],
                                                                vmax=self.vlimPx[1],
                                                                mask=self.mask,
                                                                wayBackGoodFlag=self.filters_params['wayBackGoodFlag'])

        if pr == True:

            # filters ###### important since we have a low quality level for the Good Feature to track and a high Distance Good Flag
            if self.filters_params['minNperRadius'] > 0:
                Dev, Npoints, stD = Filters.opyfFindPointsWithinRadiusandDeviation(
                    self.X, (self.V[:, 0]**2+self.V[:, 1]**2), self.filters_params['RadiusF'])
                self.X = Filters.opyfDeletePointCriterion(
                    self.X, Npoints, climmin=self.filters_params['minNperRadius'])
                self.V = Filters.opyfDeletePointCriterion(
                    self.V, Npoints, climmin=self.filters_params['minNperRadius'])
            if self.filters_params['maxDevInRadius'] != np.inf:
                Dev, Npoints, stD = Filters.opyfFindPointsWithinRadiusandDeviation(
                    self.X, self.V[:, 1], self.filters_params['RadiusF'])
                self.X = Filters.opyfDeletePointCriterion(
                    self.X, Dev, climmax=self.filters_params['maxDevInRadius'])
                self.V = Filters.opyfDeletePointCriterion(
                    self.V, Dev, climmax=self.filters_params['maxDevInRadius'])
                Dev, Npoints, stD = Filters.opyfFindPointsWithinRadiusandDeviation(
                    self.X, self.V[:, 0], self.filters_params['RadiusF'])
                self.X = Filters.opyfDeletePointCriterion(
                    self.X, Dev, climmax=self.filters_params['maxDevInRadius'])
                self.V = Filters.opyfDeletePointCriterion(
                    self.V, Dev, climmax=self.filters_params['maxDevInRadius'])
            self.scaleAndLogFlow(i)

    def scaleAndLogTracks(self, i):
        if self.scaled == True:
            self.X = (self.X-np.array(self.origin))*self.scale
            self.V = self.V*self.scale*self.fps/self.paramVecTime['step']
            self.X[:, 1] = -self.X[:, 1]
            self.V[:, 1] = -self.V[:, 1]
        print('')
        self.incr += 1
        print('-------------- [Tracking - Step '+str(self.incr)+' / ' +
              str(self.paramVecTimeTracks['Ntot'])+'] --------------')

        if self.processingMode == 'image sequence':
            print('------- From frame [' + self.listD[i-self.paramVecTimeTracks['step']
                                                      ] + '] to frame [' + self.listD[i] + '] -------')
        if self.processingMode == 'video':
            print(
                '------- From frame [' + str(i-self.paramVecTimeTracks['step']) + '] to frame [' + str(i) + '] -------')
        print('Number of Good Feature To Track = '+str(len(self.X)))
        if len(self.V) == 0:
            print(
                'No displacements measured (consider changing parameters set if displacements expected between these two frames)')
        else:
            print('Displacement max = '+str(np.max(np.absolute(self.V))) +
                  ' '+self.unit[0]+'/'+self.unit[1])

    def scaleAndLogFlow(self, i):
        if self.scaled == True:
            self.X = (self.X-np.array(self.origin))*self.scale
            self.V = self.V*self.scale*self.fps/self.paramVecTime['step']
            self.X[:, 1] = -self.X[:, 1]
            self.V[:, 1] = -self.V[:, 1]
        print('')
        self.incr += 1
        print('-------------- [Step '+str(self.incr)+' / ' +
              str(self.paramVecTime['Ntot'])+'] --------------')
        if self.processingMode == 'image sequence':
            print('------- From frame [' + self.listD[i-self.paramVecTime['step']
                                                      ] + '] to frame [' + self.listD[i] + '] -------')
        if self.processingMode == 'video':
            print(
                '------- From frame [' + str(i-self.paramVecTime['step']) + '] to frame [' + str(i) + '] -------')
        print('Number of Good Feature To Track = '+str(len(self.X)))
        if len(self.V) == 0:
            print(
                'No displacements measured (consider changing parameters set if displacements expected between these two frames)')
        else:
            print('Displacement max = '+str(np.max(np.absolute(self.V))) +
                  ' '+self.unit[0]+'/'+self.unit[1])

    def reset(self):

        self.Xdata = []
        self.Vdata = []
        self.tracks = []
        self.incr = 0
        self.UxTot = []
        self.UyTot = []
        plt.close('opyfPlot')
        self.opyfDisp = Render.opyfDisplayer(**self.paramPlot, num='opyfPlot')

    def extractGoodFeaturesAndDisplacements(self, display=False, saveImgPath=None, numberingOutput=False, imgFormat='.png', **args):
        self.reset()
        k=0
        for pr, i in zip(self.prev, self.vec):
            self.stepGoodFeaturesToTrackandOpticalFlow(pr, i)
            if pr == True:
                self.Xdata.append(self.X)
                self.Vdata.append(self.V)

                self.showXV(self.X, self.V, vis=self.vis,
                            display=display, **args)
                if saveImgPath is not None:
                    if numberingOutput==True:
                        self.opyfDisp.fig.savefig(saveImgPath+'/'+display+'_'+format(k, '04.0f')+'.'+imgFormat) 
                        k+=1
                    else: 
                        self.opyfDisp.fig.savefig(saveImgPath+'/'+display+'_'+format(
                        i, '04.0f')+'_to_'+format(i+self.paramVecTime['step'], '04.0f')+'.'+imgFormat)
                    
                if display is not False:
                    self.opyfDisp.fig.show()
                    plt.pause(0.02)

    def extractGoodFeaturesDisplacementsAndAccumulate(self, display=False, saveImgPath=None, numberingOutput=False, imgFormat='.png', **args):
        self.reset()
        self.Xaccu = np.empty((0, 2))
        self.Vaccu = np.empty((0, 2))
        k=0
        for pr, i in zip(self.prev, self.vec):
            self.stepGoodFeaturesToTrackandOpticalFlow(pr, i)
            if pr == True:
                self.Xdata.append(self.X)
                self.Vdata.append(self.V)
                self.Xaccu = np.append(self.Xaccu, self.X, axis=0)
                self.Vaccu = np.append(self.Vaccu, self.V, axis=0)
                self.showXV(self.Xaccu, self.Vaccu, vis=self.vis,
                            display=display, **args)
                if saveImgPath is not None:
                    if numberingOutput==True:
                        self.opyfDisp.fig.savefig(saveImgPath+'/'+display+'_'+format(k, '04.0f')+'.'+imgFormat) 
                        k+=1
                    else: 
                        self.opyfDisp.fig.savefig(saveImgPath+'/'+display+'_'+format(
                        i, '04.0f')+'_to_'+format(i+self.paramVecTime['step'], '04.0f')+'.'+imgFormat)
                    
                if display is not False:
                    self.opyfDisp.fig.show()
                    plt.pause(0.02)

    def interpolateOnGrid(self, X, V, mode='sequence'):

        self.interpolatedVelocities = Interpolate.npInterpolateVTK2D(
            X, V, self.XT, ParametreInterpolatorVTK=self.interp_params)

        self.gridMask[np.where(self.gridMask == 0)] = np.nan
        self.Ux = Interpolate.npTargetPoints2Grid2D(
            self.interpolatedVelocities[:, 0], self.Lgrid, self.Hgrid)*self.gridMask
        self.Uy = Interpolate.npTargetPoints2Grid2D(
            self.interpolatedVelocities[:, 1], self.Lgrid, self.Hgrid)*self.gridMask

    def extractGoodFeaturesPositionsDisplacementsAndInterpolate(self, display=False, saveImgPath=None, numberingOutput=False, Type='norme', imgFormat='png', **args):
        self.reset()
        k=0
        for pr, i in zip(self.prev, self.vec):
            self.stepGoodFeaturesToTrackandOpticalFlow(pr, i)
            if pr == True:
                self.Xdata.append(self.X)
                self.Vdata.append(self.V)
                if len(self.X) > 0:
                    self.interpolateOnGrid(self.X, self.V)
                    self.UxTot.append(np.reshape(
                        self.interpolatedVelocities[:, 0], (self.Hgrid, self.Lgrid)))
                    self.UyTot.append(np.reshape(
                        self.interpolatedVelocities[:, 1], (self.Hgrid, self.Lgrid)))
                else:
                    self.Ux, self.Uy = np.zeros((self.Hgrid, self.Lgrid)), np.zeros(
                        (self.Hgrid, self.Lgrid))*np.nan
                    self.UxTot.append(self.Ux)
                    self.UyTot.append(self.Uy)

                Field = Render.setField(self.Ux, self.Uy, Type)
                if display == 'field':
                    self.opyfDisp.plotField(Field, vis=self.vis, **args)
                    if saveImgPath is not None:                  
                        if numberingOutput==True:
                            self.opyfDisp.fig.savefig(saveImgPath+'/'+display+'_'+format(k, '04.0f')+'.'+imgFormat) 
                            k+=1
                        else: 
                            self.opyfDisp.fig.savefig(saveImgPath+'/'+display+'_'+format(
                            i, '04.0f')+'_to_'+format(i+self.paramVecTime['step'], '04.0f')+'.'+imgFormat)
                        
                    self.opyfDisp.fig.show()
                    plt.pause(0.02)
        self.fieldResults='time-serie'
        
        
    def extractGoodFeaturesDisplacementsAccumulateAndInterpolate(self, display1=False, display2=False, saveImgPath=None, Type='norme', imgFormat='png', **args):
        self.extractGoodFeaturesDisplacementsAndAccumulate(
            display=display1, **args)
        self.interpolateOnGrid(self.Xaccu, self.Vaccu)
        self.UxTot.append(np.reshape(
            self.interpolatedVelocities[:, 0], (self.Hgrid, self.Lgrid)))
        self.UyTot.append(np.reshape(
            self.interpolatedVelocities[:, 1], (self.Hgrid, self.Lgrid)))
        self.Field = Render.setField(self.Ux, self.Uy, Type)
        if display2 == 'field':

            self.opyfDisp.plotField(self.Field, vis=self.vis, **args)
            if saveImgPath is not None:
                self.opyfDisp.fig.savefig(saveImgPath+'/'+display2+'_'+format(
                    i, '04.0f')+'_to_'+format(i+self.paramVecTime['step'], '04.0f')+'.'+imgFormat)
            self.opyfDisp.fig.show()
            plt.pause(0.02)

        self.fieldResults='accumulation'
        
    def showXV(self, X, V, vis=None, display='quiver', displayColor=False, **args):
        if display == 'quiver':
            self.opyfDisp.plotQuiverUnstructured(
                X, V, vis=vis, displayColor=displayColor, **args)

        if display == 'points':
            self.opyfDisp.plotPointsUnstructured(
                Xdata=X, Vdata=V, vis=vis, displayColor=displayColor, **args)

    def scaleData(self, framesPerSecond=1, metersPerPx=1, unit=['m', 's'], origin=None):
        if self.scaled == True:
            print('datas already scaled')
        else:
            self.scaled = True

            if origin is not None:
                self.origin = origin
            else:
                self.origin = [0, self.Hvis]
            self.fps = framesPerSecond
            self.scale = metersPerPx
            self.unit = unit
            self.Time = self.Time/self.fps

            self.Vdata = [V*self.scale*self.fps /
                          self.paramVecTime['step'] for V in self.Vdata]
            self.Xdata = [(X-np.array(self.origin)) *
                          self.scale for X in self.Xdata]
            self.Ux, self.Uy = self.Ux*self.scale*self.fps / \
                self.paramVecTime['step'], self.Uy * \
                self.scale*self.fps/self.paramVecTime['step']
            self.vecX = (self.vecX-self.origin[0])*self.scale
            self.vecY = (self.vecY-self.origin[1])*self.scale

            self.grid_y, self.grid_x = (
                self.grid_y-self.origin[1])*self.scale, (self.grid_x-self.origin[0])*self.scale
            self.invertYaxis()
            self.XT = Interpolate.npGrid2TargetPoint2D(
                self.grid_x, self.grid_y)
            self.gridVx, self.gridVy = self.gridVx*self.scale*self.fps / \
                self.paramVecTime['step'], self.gridVy * \
                self.scale*self.fps/self.paramVecTime['step']

            self.interp_params = dict(Radius=self.interp_params['Radius']*self.scale,  # it is not necessary to perform interpolation on a high radius since we have a high number of values
                                      Sharpness=self.interp_params['Sharpness'],
                                      kernel='Gaussian')
            self.paramPlot = {'vecX': self.vecX,
                              'vecY': self.vecY,
                              'extentFrame': [(self.paramPlot['extentFrame'][0]-self.origin[0])*self.scale,
                                              (self.paramPlot['extentFrame']
                                               [1]-self.origin[0])*self.scale,
                                              -(self.paramPlot['extentFrame'][2]-self.origin[1])*self.scale,
                                              -(self.paramPlot['extentFrame'][3]-self.origin[1])*self.scale, ],
                              'unit': unit,
                              'vlim': [vlim*self.scale*self.fps/self.paramVecTime['step'] for vlim in self.paramPlot['vlim']]}
            plt.close('opyfPlot')
            self.opyfDisp = Render.opyfDisplayer(
                **self.paramPlot, num='opyfPlot')

    def invertYaxis(self):
        self.vecY = -self.vecY
        self.grid_y = -self.grid_y
        self.gridVx = -self.gridVx
        self.Uy = -self.Uy
        self.Xdata = [(X*np.array([1, -1])) for X in self.Xdata]
        self.Vdata = [(V*np.array([1, -1])) for V in self.Vdata]

    def writeGoodFeaturesPositionsAndDisplacements(self, fileFormat='hdf5', outFolder='.', filename=None, fileSequence=False):
        XpROI=np.copy(self.Xdata)
        if self.scaled==False:
            XpROI[:,0]=self.Xdata[:,0]+self.ROI[0]
            XpROI[:,1]=self.Xdata[:,1]+self.ROI[1]
        if filename is None:
            filename = 'good_features_positions_and_opt_flow_from_frame' + str(self.vec[0]) + 'to' + str(
                self.vec[-1]) + '_with_step_' + str(self.paramVecTime['step']) + '_and_shift_'+str(self.paramVecTime['shift'])
        if fileFormat == 'hdf5':
            Files.hdf5_WriteUnstructured2DTimeserie(
                outFolder+'/'+filename+'.'+fileFormat, self.Time, XpROI, self.Vdata)
        elif fileFormat == 'csv':
            if fileSequence == False:
                timeT = np.array(
                    [self.Time, self.Time+self.paramVecTime['step']]).T
                Files.csv_WriteUnstructured2DTimeserie2(
                    outFolder+'/'+filename+'.'+fileFormat, timeT, XpROI, self.Vdata)
            elif fileSequence == True:
                Files.mkdir2(outFolder+'/'+filename)
                for x, v, t in zip(XpROI, self.Vdata, self.Time):
                    Files.write_csvTrack2D(outFolder+'/'+filename+'/'+format(t, '04.0f')+'_to_'+format(
                        t+self.paramVecTime['step'], '04.0f')+'.'+fileFormat, x, v)

        self.writeImageProcessingParamsJSON(outFolder=outFolder)

    def writeVelocityField(self, fileFormat='hdf5', outFolder='.', filename=None, fileSequence=False, saveParamsImgProc=True):
        #for export in the image referential only if scaling and origin has not been attributed
        
        if len(self.UxTot)==0:
            sys.exit('[Warning] the following method should be run to produce an interpolated field that can be saved {extractGoodFeaturesPositionsDisplacementsAndInterpolate} or {extractGoodFeaturesDisplacementsAccumulateAndInterpolate}')

        vecXpROI=np.copy(self.vecX)
        vecYpROI=np.copy(self.vecY)
        if self.scaled==False:
            vecXpROI=self.vecX+self.ROI[0]
            vecYpROI=self.vecY+self.ROI[1]
            
        self.UxTot = np.array(self.UxTot)
        self.UyTot = np.array(self.UyTot)
        
        if self.fieldResults=='time-serie':
            if filename is None:
                filename = 'velocity_field_from_frame_' + str(self.vec[0]) + '_to_' + str(self.vec[-1]) + '_with_step_' + str(
                    self.paramVecTime['step']) + '_and_shift_'+str(self.paramVecTime['shift'])
            self.variables = [['Ux_['+self.unit[0]+'.'+self.unit[1]+'^{-1}]', self.UxTot],
                             ['Uy_['+self.unit[0]+'.'+self.unit[1]+'^{-1}]', self.UyTot]]
            if fileFormat == 'hdf5':
                Files.hdf5_Write(outFolder+'/'+filename+'.'+fileFormat, [['T_['+self.unit[0]+']', self.Time], [
                             'X_['+self.unit[0]+']', vecXpROI], ['Y_['+self.unit[0]+']', vecYpROI]], self.variables)              

            if fileFormat == 'tecplot' or 'csv' or 'tec':
                for k in range(len(self.Time)):
                    VT = Interpolate.npGrid2TargetPoint2D(self.UxTot[k], self.UyTot[k])
                    variablesTecplot = [['Ux_['+self.unit[0]+'.'+self.unit[1]+'^{-1}]', VT[:,0]],
                            ['Uy_['+self.unit[0]+'.'+self.unit[1]+'-1]', VT[:,1]]]
                    filename = 'velocity_field_from_frame_' + str(self.vec[2*k]) + '_to_' + str(self.vec[2*k+1])
                    if fileFormat == 'tecplot'  or 'tec':
                        Files.tecplot_WriteRectilinearMesh(outFolder+'/'+format(k,'04.0f')+'_'+filename+'.'+fileFormat, vecXpROI, vecYpROI, variablesTecplot)
                    if fileFormat == 'csv':
                        Files.csv_WriteRectilinearMesh(outFolder+'/'+format(k,'04.0f')+'_'+filename+'.'+fileFormat, vecXpROI, vecYpROI, variablesTecplot)
                        
        elif  self.fieldResults=='accumulation':
            VT = Interpolate.npGrid2TargetPoint2D(self.UxTot[0], self.UyTot[0])
            if filename is None:
                filename = 'velocity_field_with_accumulation_of_features_velocities_from_frame_' + str(self.vec[0]) + '_to_' + str(self.vec[-1]) + '_with_step_' + str(
                    self.paramVecTime['step']) + '_and_shift_'+str(self.paramVecTime['shift'])

            if fileFormat == 'hdf5':
                Files.hdf5_Write(outFolder+'/'+filename+'.'+fileFormat, [[
                                 'X_['+self.unit[0]+']', vecXpROI], ['Y_['+self.unit[0]+']', vecYpROI]], self.variables)
                    
            if fileFormat == 'tecplot' or 'csv' or 'tec':
                variablesTecplot = [['Ux_['+self.unit[0]+'.'+self.unit[1]+'^{-1}]', VT[:,0]],
                            ['Uy_['+self.unit[0]+'.'+self.unit[1]+'^{-1}]', VT[:,1]]]
                if fileFormat == 'tecplot'  or 'tec':
                    Files.tecplot_WriteRectilinearMesh(outFolder+'/'+filename+'.'+fileFormat, vecXpROI, vecYpROI, variablesTecplot)
                if fileFormat == 'csv':
                    Files.csv_WriteRectilinearMesh(outFolder+'/'+filename+'.'+fileFormat, vecXpROI, vecYpROI, variablesTecplot)
                    
        
        if saveParamsImgProc == True:
            self.writeImageProcessingParamsJSON(outFolder=outFolder)
            
            
            
 # TODO for csv format on file for each time step (or only one file if coordimates)
#        elif fileFormat=='csv':
#            if fileSequence==False:
#                Files.csv_WriteUnstructured2DTimeserie(outFolder+'/'+filename+'.'+fileFormat,self.Time,self.Xdata,self.Vdata)
#            elif fileSequence==True:
#                Files.mkdir2(outFolder+'/'+filename)
#                for x,v,t in zip(self.Xdata,self.Vdata,self.Time):
#                    Files.write_csvTrack2D(outFolder+'/'+filename+'/'+format(t,'04.0f')+'_to_'+format(t+self.paramVecTime['step'],'04.0f')+'.'+fileFormat,x,v)
#
#

    def writeTracks(self, filename=None, fileFormat='csv', outFolder='.'):
        import csv
        print('')

        TL = self.tracks_params['track_len']
        B = 0
        if TL < self.paramVecTimeTracks['Ntot']:
            print('Track_length has been set at [' + str(
                TL) + '] and step at [' + str(self.paramVecTimeTracks['step']) + ']')
            print('Consequently, and considering the tracking processing plan, no tracks before frame {last_frame - Track_length*step}, i.e. '+str(
                self.vecTracks[-1]-(TL-1)*self.paramVecTimeTracks['step'])+' will be saved')
            B = TL
        if filename is None:
            filename = 'tracks_from_frame' + str(self.vecTracks[-B]) + 'to' + str(self.vecTracks[-1]) + '_with_step_' + str(
                self.paramVecTimeTracks['step'])
        f = open(filename, 'w')
        writer = csv.DictWriter(
            f, fieldnames=['track_index', 'frame_index', 'X', 'Y', 'Vx', 'Vy'])
        writer.writeheader()
        N = 0

        for (tr, vtr) in zip(self.tracks, self.vtracks):
            l = len(tr)
            for i in range(l):
                if self.scaled==True:
                    writer.writerow(
                    {'track_index': N, 'frame_index': self.vecTracks[-l+i], 'X': tr[i][0], 'Y': tr[i][1], 'Vx': vtr[i][0], 'Vy': vtr[i][1]})
                else:
                    writer.writerow(
                    {'track_index': N, 'frame_index': self.vecTracks[-l+i], 'X': tr[i][0]+self.ROI[0], 'Y': tr[i][1]+self.ROI[1], 'Vx': vtr[i][0], 'Vy': vtr[i][1]})

            N += 1
        f.close()

    def writeImageProcessingParamsJSON(self, outFolder='.', filename=None):

        fulldict = {'lk_params': self.lk_params, 'feature_params': self.feature_params,
                    'filters_params': self.filters_params, 'interp_parameter': self.interp_params,
                    'vecTime_params':self.paramVecTime}
        fulldict['fieldResults']=self.fieldResults
        Files.writeImageProcessingParamsJSON(
            fulldict, outFolder=outFolder, filename=None)


#            for x,v in zip(Xdata,Vdata):

    # def extractGoodFeaturesPositionsAndExtractFeatures(self, display=False, windowSize=(16, 16)):

    #     for pr, i in zip(self.prev, self.vec):

    #         #    frame=frame[:,:,1]
    #         l = self.listD[i]

    #         vis = cv2.imread(self.folder_src+'/'+l)  # special for tiff images

    #         vis = Render.CLAHEbrightness(
    #             vis, 0, tileGridSize=(20, 20), clipLimit=2)
    #         if len(np.shape(vis)) == 3:
    #             current_gray = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)
    #         else:
    #             current_gray = vis
    #         current_gray = current_gray[self.ROI[1]:(
    #             self.ROI[3]+self.ROI[1]), self.ROI[0]:(self.ROI[2]+self.ROI[0])]

    #         p0 = cv2.goodFeaturesToTrack(current_gray, **self.feature_params)
    #         X = p0.reshape(-1, 2)
    #         samples = []
    #         for x in X:
    #             featureExtraction = current_gray[int(x[0]-windowSize[0]/2):int(
    #                 x[0]+windowSize[0]/2), int(x[1]-windowSize[1]/2):int(x[1]+windowSize[1]/2)]
    #             samples.append(featureExtraction)

    #         self.samplesMat.append(samples)
    #         self.Xsamples.append(X)
    #         if display == 'points':
    #             self.show(self, X, X, vis, display='points')


class videoAnalyzer(Analyzer):
    def __init__(self, video_src, **args):
        self.processingMode = 'video'
        self.video_src = video_src
        self.cap = cv2.VideoCapture(self.video_src)
        self.ret, self.frameInit = self.cap.read()
        if self.ret == False:
            print('Error: the video file path might be wrong')
            sys.exit()
        self.frameInit = Tools.convertToGrayScale(self.frameInit)
        self.number_of_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        Analyzer.__init__(self, **args)


class frameSequenceAnalyzer(Analyzer):
    def __init__(self,folder_src, **args):
        self.processingMode = 'image sequence'
        self.folder_src = folder_src
        self.listD = np.sort(os.listdir(self.folder_src))
        self.number_of_frames = len(self.listD)
        self.frameInit = cv2.imread(self.folder_src+'/'+self.listD[0])
        self.frameInit = Tools.convertToGrayScale(self.frameInit)

        Analyzer.__init__(self, **args)

#    def writeGoodFeaturesPositionsAndExtraction(self,fileFormat='hdf5',imgFormat='png',outFolder='',filename=None,fileSequence=False):
#        if filename is None:
#            filename='Goof_features_positions_from_frame'+ str(self.vec[0]) + 'to' + str(self.vec[-2]) + '_with_shift_'+str(self.paramVecTime['shift'])
#
#        for samples in self.samplesMat:
#           folderFrame=outFolder+'/'+filename+'/'+format(t,'04.0f')+'_to_'+format(t+self.paramVecTime['step']
#           Files.mkdir2()
#           cv2.imwrite()
