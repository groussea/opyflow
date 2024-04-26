#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 08:59:41 2019

@author: Gauthier ROUSSEAU
"""
#%%
import os
import sys
import numpy as np
import cv2
from opyf import MeshesAndTime, Track, Render, Files, Tools, Interpolate, Filters
import matplotlib.pyplot as plt
import time
import opyf
import matplotlib as mpl
import re
#%%

class Analyzer():
    def __init__(
            self,
            imageROI=None,
            num='opyfPlot',
            mute=False,
            close_at_reset=True,
            mask=None,
            display=True,
            frameInit=None,
            number_of_frames=0,
            processingMode='',
            listD=[],
            folder_src='',
            cap=None,
            **args):
        self.cap = cap
        self.limitRAMinMByte = 2000
        self.processingMode = processingMode
        self.listD = listD
        self.folder_src = folder_src
        self.frameInit = frameInit
        self.number_of_frames = number_of_frames
        print('Dimensions :\n \t', 'Width :',
              self.frameInit.shape[1], 'Height :', self.frameInit.shape[0])
        self.num = num
        self.display = display
        self.mute = mute
        if self.display is False:
            self.mute = True
        self.scaled = False
        self.scale = 1
        if imageROI is None:
            self.ROI = [0, 0, self.frameInit.shape[1], self.frameInit.shape[0]]
        else:
            self.ROI = imageROI
        print('Regio Of Interest :\n \t', self.ROI)
        self.cropFrameInit = self.frameInit[self.ROI[1]:(
            self.ROI[3] + self.ROI[1]), self.ROI[0]:(self.ROI[2] + self.ROI[0])]
        self.Hvis, self.Lvis = self.cropFrameInit.shape

        self.set_mask(mask)

        self.invertedXY = False

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
        self.paramPlot = {
            'ScaleVectors': args.get(
            'ScaleVectors', 0.1), 'vecX': [], 'vecY': [], 
            'extentFrame': args.get('extentFrame', [0, self.Lvis, self.Hvis, 0]), 
            'unit': args.get('unit', ['px', 'deltaT']), 'Hfig': args.get('Hfig', 8),
            'force_figsize': args.get('force_figsize', None),'grid': args.get( 'grid', True), 
            'vlim': args.get('vlim', [ 0, 40])}
        self.birdEyeMod = False
        self.stabilizeOn = False

        self.close_at_reset = close_at_reset
        self.reset(first=True)
        self.set_gridToInterpolateOn()

        self.set_vecTime()
        print('number of frames : ' + str(self.number_of_frames))
        self.fieldResults = None
        # plt.ion()
        self.visInit = np.copy(self.vis)

        self.dumpShow()

        """
        loading an optional mask
        the mask entry may be the path to the image mask or directly a numpy.array
        It can be either of type boolean or uint8.
        """

    def set_mask(self, mask):
        self.mask = mask
        if isinstance(mask, str):
            self.mask = cv2.imread(mask)
            if self.mask is None:
                print('Path to the Mask is wrong')

        if isinstance(self.mask, np.ndarray):
            if self.mask.dtype == bool:
                if len(self.mask.shape == 3):
                    self.mask = self.mask[:, :, 0]
                else:
                    self.mask = self.mask
            else:
                self.mask = Tools.convertToGrayScale(self.mask) > 100
        else:
            self.mask = np.ones_like(self.cropFrameInit)

    def invertXYlabel(self):
        if self.display:
            self.invertedXY = bool(self.invertedXY - 1)
            self.opyfDisp.invertXYlabel()

    def dumpShow(self):
        if self.display and self.mute is False:
            self.opyfDisp.ax.imshow(cv2.cvtColor(self.vis, cv2.COLOR_BGR2RGB))
            if plt.rcParams['backend'] in mpl.rcsetup.interactive_bk:
                self.opyfDisp.fig.show()
            time.sleep(0.1)

    def set_goodFeaturesToTrackParams(
            self,
            maxCorners=40000,
            qualityLevel=0.005,
            minDistance=5,
            blockSize=16):

        self.feature_params = dict(maxCorners=maxCorners,
                                   qualityLevel=qualityLevel,
                                   minDistance=minDistance,
                                   blockSize=blockSize)
        print('')
        print('Good Features To Track Parameters:')
        for x in self.feature_params:
            print('\t- ', x, ':', self.feature_params[x])

    def set_opticalFlowParams(
        self, winSize=(
            16, 16), maxLevel=3, criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.03)):

        self.lk_params = dict(winSize=winSize,
                              maxLevel=maxLevel,
                              criteria=criteria)
        print('')
        print('Optical Flow Parameters:')
        for x in self.lk_params:
            print('\t- ', x, ':', self.lk_params[x])
    # for the moment the radius must be given in px (more intuitive but should
    # be clearer)

    def set_filtersParams(self,
                          RadiusF=30,
                          minNperRadius=0,
                          maxDevInRadius=np.inf,
                          wayBackGoodFlag=np.inf,
                          CLAHE=False,
                          range_Vx=[-np.inf,
                                    np.inf],
                          range_Vy=[-np.inf,
                                    np.inf],
                          clplim=3,
                          gridSize=(20,
                                    20)):

        self.filters_params = dict(RadiusF=RadiusF,
                                   minNperRadius=minNperRadius,
                                   maxDevInRadius=maxDevInRadius,
                                   wayBackGoodFlag=wayBackGoodFlag,
                                   CLAHE=CLAHE,
                                   range_Vx=range_Vx,
                                   range_Vy=range_Vy,
                                   clplim=clplim,
                                   gridSize=gridSize)

        print('')
        print('Filters Params:')
        for x in self.filters_params:
            print('\t- ', x, ':', self.filters_params[x])

    # When the data set is scaled the radius must be gven in the good length
    # unity (meter, cm, ....) 
    def set_interpolationParams(
            self,
            Radius=None,
            Sharpness=8,  
            kernel='Gaussian'):
        if Radius is None:
            Radius = self.scale * 30  # 30 px when the data set is not scaled
        else:
            Radius = Radius
    # set the a scaled value if the user does not introduce a Radius and the
    # data set is scaled
        self.interp_params = dict(Radius=Radius,  
                                      Sharpness=Sharpness,
                                      kernel=kernel)
        print('')
        print('Interpolation Parameters:')
        for x in self.interp_params:
            print('\t- ', x, ':', self.interp_params[x])

    def set_vlim(self, vlim):
        if not self.scaled:
            self.vlimPx = vlim
        else:
            self.vlimPx = [v / (self.scale * self.fps /
                                self.paramVecTime['step']) for v in vlim]

        self.paramPlot['vlim'] = vlim
        if self.display:
            self.opyfDisp.paramPlot['vlim'] = vlim

        print('Velocity limits: ', vlim[0])
        print('\t minimum norm velocity: ', vlim[0])
        print('\t maximum norm velocity: ', vlim[1])

    def set_gridToInterpolateOn(
            self,
            pixLeft=0,
            pixRight=0,
            stepHor=2,
            pixUp=0,
            pixDown=0,
            stepVert=2,
            ROI=None,
            stepGrid=None):
        if pixRight == 0:
            pixRight = self.Lvis
        if pixDown == 0:
            pixDown = self.Hvis
        if stepGrid is not None:
            stepVert, stepHor = stepGrid, stepGrid

        if ROI is not None:
            self.ROImeasure = ROI
            pixLeft, pixRight, pixUp, pixDown = ROI[0], ROI[0] + \
                ROI[2], ROI[1], ROI[1] + ROI[3]
        else:
            self.ROImeasure = [
                pixLeft,
                pixUp,
                pixRight - pixLeft,
                pixDown - pixUp]
        self.grid_y, self.grid_x, self.gridVx, self.gridVy, self.Hgrid, self.Lgrid = MeshesAndTime.set_gridToInterpolateOn(
            pixLeft, pixRight, stepHor, pixUp, pixDown, stepVert)

        if self.scaled:
            self.grid_y, self.grid_x = - \
                (self.grid_y - self.origin[1]) * \
                self.scale, (self.grid_x - self.origin[0]) * self.scale

        self.vecX = self.grid_x[0, :]
        self.vecY = self.grid_y[:, 0]
        self.XT = Interpolate.npGrid2TargetPoint2D(self.grid_x, self.grid_y)
        self.paramPlot['vecX'] = self.vecX
        self.paramPlot['vecY'] = self.vecY
        if self.display:
            self.opyfDisp.paramPlot['vecX'] = self.vecX
            self.opyfDisp.paramPlot['vecY'] = self.vecY
            self.opyfDisp.reset()
        self.Ux, self.Uy = np.zeros((self.Hgrid, self.Lgrid)), np.zeros(
            (self.Hgrid, self.Lgrid))
        self.gridMask = cv2.resize(
            np.uint8(self.mask), (self.Lgrid, self.Hgrid)) > 0.5

    def set_imageParams(self, **args):
        self.rangeOfPixels = args.get('rangeOfPixels', [0, 255])
        self.imreadOption = args.get('imreadOption', 1)
        if self.imreadOption == 'ANYDEPTH':
            self.imreadOption = 2

    def set_vecTime(self, starting_frame=0, step=1, shift=1, Ntot=1):
        self.paramVecTime = {'starting_frame': starting_frame,
                             'step': step,
                             'shift': shift,
                             'Ntot': Ntot}

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
        file_prev = ''
        if self.processingMode == 'image sequence' and self.mute == False:
            for pr, i in zip(self.prev, self.vec):
                if not pr:
                    print(
                        '--> detect Good Features to Track on image [' + self.listD[i] + ']')
                    file_prev = self.listD[i]
                else:
                    print('--> diplacements measurement between images [' +file_prev +
                        '] and [' +self.listD[i] +']')

        if self.processingMode == 'video' and self.mute == False:
            for pr, i in zip(self.prev, self.vec):
                if not pr:
                    print( '--> detect Good Features to Track on frame [' + str(i) + ']')
                    file_prev = str(i)
                else:
                    print('--> diplacements measurement between frame [' +
                        file_prev +'] and [' +str(i) +']')
        self.Time = (self.vec[0:-1:2] + self.vec[1::2]) / 2
        # initilize self.vis with the first frame of the set

        if self.processingMode == 'video':
            self.loadDict(0)
        self.readFrame(self.vec[0])
        self.dumpShow()

    def loadDict(self, iL):
        self.dictFrames = {}
        self.sortedVec = np.sort(np.unique(self.vec))
        self.cap.set(
            cv2.CAP_PROP_POS_FRAMES, max(
                0, iL - self.paramVecTime['step']))
        k = 0
        while self.sortedVec[k] < iL:
            k += 1
        self.totDictByte = 0
        vis=0
        while k < len(
                self.sortedVec) and self.totDictByte < self.limitRAMinMByte and vis is not None:
            indF = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            ret, vis = self.cap.read()
            if indF == self.sortedVec[k]:
                self.dictFrames[str(self.sortedVec[k])] = vis
                k += 1
                if vis is not None:
                    self.totDictByte += sys.getsizeof(vis.tobytes()) / 1e6

    def set_trackingFeatures(
            self,
            starting_frame=0,
            step=1,
            Ntot=10,
            track_length=10,
            detection_interval=10):

        self.tracks_params = {'track_len': track_length,
                              'detect_interval': detection_interval}

        self.paramVecTimeTracks = {'starting_frame': starting_frame,
                                   'step': step,
                                   'Ntot': Ntot}

        self.vecTracks, self.prevTracks = MeshesAndTime.set_vecTimeTracks(
            starting_frame=starting_frame, step=step,  Ntot=Ntot)

        print(
            '\nTracking processing \nTracking operates only in succesives images sperated with [step] frames')
        if self.vecTracks[-1] > self.number_of_frames:
            print('----- Error ----')
            print('Your tracking plan is not compatible with the frame set')
            print(
                'Consider that for tracking [starting_frame+step*(Ntot)]  must be smaller than the number of frames')
            sys.exit()

        print('Starting frame : [' + str(starting_frame) + ']')

        print('--> Tracking operate from frame [' + str(
            self.vecTracks[1]) + '] to [' + str(self.vecTracks[-1]) + ']')

        print('On the first frame, only Good Features to Track are detected')

        print('Maximum tracking length are [' + str(track_length) + ']')
        print(
            'Good features detection are reloaded every [' +
            str(detection_interval) +
            '] frames')
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
        self.readFrame(self.vecTracks[0])

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
            self.vis = cv2.imread(self.folder_src + '/' + l,
                                  self.imreadOption)  # 2 for tiff images
            if self.vis is None:
                print('')
                sys.exit("Impossible to read the frame " + l)
        elif self.processingMode == 'video':
            try:
                self.vis = self.dictFrames[str(i)]
            except BaseException:
                self.loadDict(i)
                self.vis = self.dictFrames[str(i)]

        self.vis = self.vis[self.ROI[1]:(
            self.ROI[3] + self.ROI[1]), self.ROI[0]:(self.ROI[2] + self.ROI[0])]

        if self.rangeOfPixels != [0, 255]:
            self.vis = np.array(self.vis, dtype='float32')
            self.vis = ((self.vis - self.rangeOfPixels[0]) / (
                self.rangeOfPixels[1] - self.rangeOfPixels[0])) * 255
            self.vis[np.where(self.vis < 0)] = 0
            self.vis[np.where(self.vis > 255)] = 255
            self.vis = np.uint16(self.vis)
        if self.stabilizeOn:
            self.stabilize(i)
        if self.birdEyeMod:
            self.transformBirdEye()

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

    # def extractDeepFlow(self, display=False, saveImgPath=None, Type='norm', imgFormat='png', **args):
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
    #                 time.sleep(0.1)

    def stepTracks(self, pr, i):
        if not pr:
            self.prev_gray = None
        self.currentFrame = i
        self.readFrame(self.currentFrame)
        self.substractAveragedFrame()
        self.tracks, self.vtracks, self.prev_gray, self.X, self.V = Track.opyfTrack(self.tracks, self.vtracks, self.gray, self.prev_gray, self.incr, self.feature_params, self.lk_params, self.tracks_params, ROI=self.ROImeasure, vmin=self.vlimPx[0], vmax=self.vlimPx[1], mask=self.mask, wayBackGoodFlag=self.filters_params['wayBackGoodFlag'])
        self.scaleAndLogTracks(i)

    def extractTracks(
            self,
            display=False,
            saveImgPath=None,
            numberingOutput=False,
            imgFormat='png',
            **args):
        self.reset()
        self.tracks = []
        self.vtracks = []
        if self.prevTracks is None:
            print( '\n \nWARNING : To run the extractTracks() method, it is mandatory to define the tracking plan through the method [set_trackingFeatures()]\n\n')
            sys.exit()
        k = 0
        for pr, i in zip(self.prevTracks, self.vecTracks):
            self.stepTracks(pr, i)
            if pr:
                self.Xdata.append(self.X)
                self.Vdata.append(self.V)

                cv2.polylines(self.vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
                self.showXV(self.X, self.V, vis=self.vis,
                            display=display, **args)
                if saveImgPath is not None and self.display:
                    if numberingOutput:
                        self.opyfDisp.fig.savefig(
                            str(saveImgPath + '/' + display + '_' + format(k, '04.0f') + '.' + imgFormat))
                        k += 1
                    else:
                        self.opyfDisp.fig.savefig(str(saveImgPath + '/' + display + '_' + format(i, '04.0f') +
                        '_to_' + format(i + self.paramVecTime['step'], '04.0f') + '.' + imgFormat))

                if display is not False:
                    # self.opyfDisp.fig.show()
                    time.sleep(0.1)

    def substractAveragedFrame(self):
        self.gray = Tools.convertToGrayScale(self.vis)
        if self.frameAv is not None:
            self.gray = np.array(self.gray, dtype='float32')
            self.gray = self.gray - self.frameAv
            self.gray[np.where(self.gray < 0)] = 0
            self.gray[np.where(self.gray > 255)] = 255
            self.gray = np.array(self.gray, dtype='uint8')

    def stepGoodFeaturesToTrackandOpticalFlow(self, pr, i):
        if not pr:
            self.prev_gray = None
        self.readFrame(i)
        self.substractAveragedFrame()

        if self.filters_params['CLAHE']:
            self.gray = Render.CLAHEbrightness(
                self.gray, 0,
                tileGridSize=self.filters_params['gridSize'],
                clipLimit=self.filters_params['clplim'])
#             self.vis=Render.CLAHEbrightness(self.vis,0,tileGridSize=(20,20),clipLimit=2)

        self.prev_gray, self.X, self.V = Track.opyfFlowGoodFlag(self.gray, self.prev_gray, self.feature_params, self.lk_params, ROI=self.ROImeasure, vmin=self.vlimPx[0],  vmax=self.vlimPx[1], mask=self.mask, wayBackGoodFlag=self.filters_params['wayBackGoodFlag'])

        if pr:
            # filters ######
            self.scaleAndLogFlow(i)
            self.X, self.V = self.applyFilters(self.X, self.V)

    def applyFilters(self, X, V):
        if self.filters_params['minNperRadius'] > 0:
            print('[I] Number of point within radius (in px) filter')
            Dev, Npoints, stD = Filters.opyfFindPointsWithinRadiusandDeviation(
                X, (V[:, 0]**2 + V[:, 1]**2), self.filters_params['RadiusF'])
            X = Filters.opyfDeletePointCriterion(
                X, Npoints, climmin=self.filters_params['minNperRadius'])
            V = Filters.opyfDeletePointCriterion(
                V, Npoints, climmin=self.filters_params['minNperRadius'])
        if self.filters_params['maxDevInRadius'] != np.inf:
            print('[I] Deviation filter')
            print('[I] Step vertical')
            Dev, Npoints, stD = Filters.opyfFindPointsWithinRadiusandDeviation(
                X, V[:, 1], self.filters_params['RadiusF'])
            X = Filters.opyfDeletePointCriterion(
                X, Dev, climmax=self.filters_params['maxDevInRadius'])
            V = Filters.opyfDeletePointCriterion(
                V, Dev, climmax=self.filters_params['maxDevInRadius'])
            print('[I] Step horizontal')
            Dev, Npoints, stD = Filters.opyfFindPointsWithinRadiusandDeviation(
                X, V[:, 0], self.filters_params['RadiusF'])
            X = Filters.opyfDeletePointCriterion(
                X, Dev, climmax=self.filters_params['maxDevInRadius'])
            V = Filters.opyfDeletePointCriterion(
                V, Dev, climmax=self.filters_params['maxDevInRadius'])
        if self.filters_params['range_Vx'][0] != - \
                np.inf or self.filters_params['range_Vx'][1] != np.inf:
            inDel = []
            for x, v, k in zip(X, V, range(len(V))):
                if v[0] < self.filters_params['range_Vx'][0] or v[0] > self.filters_params['range_Vx'][1]:
                    inDel.append(k)
            X = np.delete(X, inDel, 0)
            V = np.delete(V, inDel, 0)
        if self.filters_params['range_Vy'][0] != - \
                np.inf or self.filters_params['range_Vy'][1] != np.inf:
            inDel = []
            for x, v, k in zip(X, V, range(len(V))):
                if v[1] < self.filters_params['range_Vy'][0] or v[1] > self.filters_params['range_Vy'][1]:
                    inDel.append(k)
            X = np.delete(X, inDel, 0)
            V = np.delete(V, inDel, 0)
        return X, V

    def scaleAndLogTracks(self, i):
        if self.scaled and len(self.X) > 0:
            self.X = (self.X - np.array(self.origin)) * self.scale
            self.V = self.V * self.scale * self.fps / self.paramVecTime['step']
            self.X[:, 1] = -self.X[:, 1]
            self.V[:, 1] = -self.V[:, 1]
        print('')
        self.incr += 1
        print('-------------- [Tracking - Step ' + str(self.incr) + ' / ' +
              str(self.paramVecTimeTracks['Ntot']) + '] --------------')

        if self.processingMode == 'image sequence':
            print('------- From frame [' + self.listD[i - self.paramVecTimeTracks['step']] + '] to frame [' + self.listD[i] + '] -------')
        if self.processingMode == 'video':
            print('------- From frame [' + str(i -  self.paramVecTimeTracks['step']) + '] to frame [' + str(i) + '] -------')
        print('Number of Good Feature To Track = ' + str(len(self.X)))
        if len(self.V) == 0:
            print(
                'No displacements measured (consider changing parameters set if displacements expected between these two frames)')
        else:
            print('Displacement max = ' + str(np.max(np.absolute(self.V))) +  ' ' + self.unit[0] + '/' + self.unit[1])

    def scaleAndLogFlow(self, i):
        if self.scaled:
            self.X = (self.X - np.array(self.origin)) * self.scale
            self.V = self.V * self.scale * self.fps / self.paramVecTime['step']
            self.X[:, 1] = -self.X[:, 1]
            self.V[:, 1] = -self.V[:, 1]
        print('')
        self.incr += 1
        print('-------------- [Step ' + str(self.incr) + ' / ' +
              str(self.paramVecTime['Ntot']) + '] --------------')
        if self.processingMode == 'image sequence':
            print('------- From frame [' +
                  self.listD[i - self.paramVecTime['step']] + '] to frame [' + self.listD[i] + '] -------')
        if self.processingMode == 'video':
            print('------- From frame [' + str(i - self.paramVecTime['step']) +  '] to frame [' + str(i) + '] -------')
        print('Number of Good Feature To Track = ' + str(len(self.X)))
        if len(self.V) == 0:
            print(
                'No displacements measured (consider changing parameters set if displacements expected between these two frames)')
        else:
            print('Displacement max = ' + str(np.max(np.absolute(self.V))) +
                  ' ' + self.unit[0] + '/' + self.unit[1])

    def reset(self, first=False):
        self.Xdata = []
        self.Vdata = []
        self.tracks = []
        self.incr = 0
        self.UxTot = []
        self.UyTot = []
        self.Xaccu = np.empty((0, 2))
        self.Vaccu = np.empty((0, 2))
        if (self.close_at_reset or first) and self.display:
            plt.close(self.num)
            self.opyfDisp = Render.opyfDisplayer(
                **self.paramPlot, num=self.num)
            if self.invertedXY:
                self.opyfDisp.invertXYlabel()
            if plt.rcParams['backend'] in mpl.rcsetup.interactive_bk:
                self.opyfDisp.fig.show()

    def extractGoodFeaturesAndDisplacements(
            self,
            display='quiver',
            saveImgPath=None,
            numberingOutput=False,
            imgFormat='.png',
            **args):

        self.reset()
        k = 0
        for pr, i in zip(self.prev, self.vec):
            self.stepGoodFeaturesToTrackandOpticalFlow(pr, i)
            if pr:
                self.Xdata.append(self.X)
                self.Vdata.append(self.V)

                self.showXV(self.X, self.V, vis=self.vis,
                            display=display, **args)
                if saveImgPath is not None and self.display:
                    if numberingOutput:
                        self.opyfDisp.fig.savefig(
                            str(saveImgPath + '/' + display + '_' + format(k, '04.0f') + '.' + imgFormat))
                        k += 1
                    else:
                        self.opyfDisp.fig.savefig(str(saveImgPath + '/' + display + '_' + format(i, '04.0f') +
                        '_to_' + format(i + self.paramVecTime['step'], '04.0f') + '.' + imgFormat))

                if display is not False:
                    # self.opyfDisp.fig.show()
                    time.sleep(0.1)

    def extractGoodFeaturesDisplacementsAndAccumulate(
            self,
            display='quiver',
            saveImgPath=None,
            numberingOutput=False,
            imgFormat='.png',
            **args):
        self.reset()
        self.Xaccu = np.empty((0, 2))
        self.Vaccu = np.empty((0, 2))
        k = 0
        for pr, i in zip(self.prev, self.vec):
            self.stepGoodFeaturesToTrackandOpticalFlow(pr, i)
            if pr:
                self.Xdata.append(self.X)
                self.Vdata.append(self.V)
                self.Xaccu = np.append(self.Xaccu, self.X, axis=0)
                self.Vaccu = np.append(self.Vaccu, self.V, axis=0)
                self.showXV(self.Xaccu, self.Vaccu, vis=self.vis,
                            display=display, **args)
                if saveImgPath is not None and self.display:
                    if numberingOutput:
                        self.opyfDisp.fig.savefig(
                            str(saveImgPath + '/' + display + '_' + format(k, '04.0f') + '.' + imgFormat))
                        k += 1
                    else:
                        self.opyfDisp.fig.savefig(str(saveImgPath + '/' + display + '_' + format(i, '04.0f') +
                        '_to_' + format(i + self.paramVecTime['step'], '04.0f') + '.' + imgFormat))

                if display is not False:
                    # self.opyfDisp.fig.show()
                    time.sleep(0.1)

    def interpolateOnGrid(self, X, V):
        print('[I] ' + str(len(X)) + ' vectors to interpolate')
        self.interpolatedVelocities = Interpolate.npInterpolateVTK2D(
            X, V, self.XT, ParametreInterpolatorVTK=self.interp_params)

        # self.gridMask[np.where(self.gridMask == 0)] = np.nan
        self.Ux = Interpolate.npTargetPoints2Grid2D(
            self.interpolatedVelocities[:, 0], self.Lgrid, self.Hgrid) * self.gridMask
        self.Uy = Interpolate.npTargetPoints2Grid2D(
            self.interpolatedVelocities[:, 1], self.Lgrid, self.Hgrid) * self.gridMask

    def extractGoodFeaturesPositionsDisplacementsAndInterpolate(
            self,
            display='field',
            saveImgPath=None,
            numberingOutput=False,
            Type='norm',
            imgFormat='png',
            **args):
        self.reset()
        k = 0
        for pr, i in zip(self.prev, self.vec):
            self.stepGoodFeaturesToTrackandOpticalFlow(pr, i)
            if pr:
                self.Xdata.append(self.X)
                self.Vdata.append(self.V)
                if len(self.X) > 0:
                    self.interpolateOnGrid(self.X, self.V)
                    self.UxTot.append(np.reshape(
                        self.interpolatedVelocities[:, 0], (self.Hgrid, self.Lgrid)))
                    self.UyTot.append(np.reshape(
                        self.interpolatedVelocities[:, 1], (self.Hgrid, self.Lgrid)))
                else:
                    self.Ux, self.Uy = np.zeros(
                        (self.Hgrid, self.Lgrid)), np.zeros(
                        (self.Hgrid, self.Lgrid)) * np.nan
                    self.UxTot.append(self.Ux)
                    self.UyTot.append(self.Uy)
                    print(
                        'no velocites on Good Features have been found during this step')

                self.Field = Render.setField(self.Ux, self.Uy, Type)
                self.Field[np.where(self.Field == 0)] = np.nan
                self.Field = self.Field * self.gridMask
                if display == 'field' and self.mute == False and self.display == True:
                    self.opyfDisp.plotField(self.Field, vis=self.vis, **args)
                    time.sleep(0.1)
                elif display == 'quiver_on_field' and self.mute == False and self.display == True:
                    self.opyfDisp.plotQuiverField(
                        self.Ux, self.Uy, vis=self.vis, **args)
                if (saveImgPath is not None) * (display ==
                                                'field' or display == 'quiver_on_field'):
                    if numberingOutput:
                        fileN = str(saveImgPath + '/' + display + '_' +
                            format(k, '04.0f') + '.' + imgFormat)
                        self.opyfDisp.fig.savefig(fileN)
                        k += 1
                    else:
                        fileN = str(saveImgPath + '/' + display + '_' + format(i, '04.0f') +
                        '_to_' + format(i + self.paramVecTime['step'], '04.0f') + '.' + imgFormat)
                        self.opyfDisp.fig.savefig(fileN)
        self.fieldResults='time-serie'

    def extractGoodFeaturesDisplacementsAccumulateAndInterpolate(
            self,
            display1 = 'quiver',
            display2 = 'field',
            saveImgPath = None,
            Type = 'norm',
            imgFormat = 'png',
            **args):
        self.imgFormat = imgFormat
        self.extractGoodFeaturesDisplacementsAndAccumulate(
            display = display1, **args)
        self.interpolateOnGrid(self.Xaccu, self.Vaccu)
        self.UxTot.append(np.reshape(
            self.interpolatedVelocities[:, 0], (self.Hgrid, self.Lgrid)))
        self.UyTot.append(np.reshape(
            self.interpolatedVelocities[:, 1], (self.Hgrid, self.Lgrid)))
        self.Field=Render.setField(self.Ux, self.Uy, Type)
        self.Field[np.where(self.Field == 0)]=np.nan
        self.Field=self.Field * self.gridMask
        if display2 == 'field' and self.mute == False and self.display == True:
            self.opyfDisp.plotField(self.Field, vis = self.vis, **args)
            if saveImgPath is not None:
                self.opyfDisp.fig.savefig(saveImgPath + '/' + display2 + '_' + format(
                    self.vec[0], '04.0f') + '_to_' + format(self.vec[-1], '04.0f') + '.' + imgFormat)
            # self.opyfDisp.fig.show()
            time.sleep(0.1)

        self.fieldResults = 'accumulation'

    def filterAndInterpolate(
            self,
            Type='norm',
            display='field',
            saveImgPath=None,
            **args):
        self.Xaccu, self.Vaccu = self.applyFilters(self.Xaccu, self.Vaccu)
        self.interpolateOnGrid(self.Xaccu, self.Vaccu)
        self.UxTot = []
        self.UyTot = [] 
        self.UxTot.append(np.reshape(
            self.interpolatedVelocities[:, 0], (self.Hgrid, self.Lgrid)))
        self.UyTot.append(np.reshape(
            self.interpolatedVelocities[:, 1], (self.Hgrid, self.Lgrid)))
        self.Field=Render.setField(self.Ux, self.Uy, Type)
        self.Field[np.where(self.Field == 0)]=np.nan
        self.Field=self.Field * self.gridMask

        self.fieldResults = 'accumulation'

    def showXV(self, X, V, vis = None,
            display = 'quiver', displayColor = True, **args):
        if display == 'quiver' and self.mute == False and self.display == True:
            self.opyfDisp.plotQuiverUnstructured(
                X, V, vis = vis, displayColor = displayColor, **args)

        if display == 'points' and self.mute == False and self.display == True:
            self.opyfDisp.plotPointsUnstructured(
                Xdata = X, Vdata = V, vis = vis, displayColor = displayColor, **args)

    def scaleData( self, framesPerSecond = None,
            metersPerPx = None,
            unit = [ 'm', 's'], origin = None):

        if self.scaled:
            print('datas already scaled')
        else:
            self.scaled=True

            if origin is not None:
                self.origin=origin
            else:
                self.origin=[self.ROI[0], self.ROI[1]]
            self.fps=framesPerSecond
            self.scale=metersPerPx
            self.unit=unit
            self.Time=self.Time / self.fps
            if hasattr(self, 'X'):
                self.X=(self.X - np.array(self.origin)) * self.scale
                self.V=self.V * self.scale *self.fps / self.paramVecTime['step']
                self.Vdata=[V * self.scale * self.fps /
                              self.paramVecTime['step'] for V in self.Vdata]

                self.Xdata = [(X - np.array(self.origin)) *
                              self.scale for X in self.Xdata]
            if hasattr(self, 'Ux'):
                self.Ux, self.Uy= self.Ux * self.scale * self.fps / self.paramVecTime['step'], self.Uy * self.scale * self.fps / self.paramVecTime['step']
                self.vecX = (self.vecX - self.origin[0]) * self.scale
                self.vecY = (self.vecY - self.origin[1]) * self.scale
                self.gridVx, self.gridVy= self.gridVx * self.scale * self.fps / self.paramVecTime['step'], self.gridVy *self.scale * self.fps / self.paramVecTime['step']
            if hasattr(self, 'UyTot'):
                for i in range(len(self.UyTot)):
                    self.UyTot[i]= self.UyTot[i] * self.scale *   self.fps / self.paramVecTime['step']
                    self.UxTot[i]= self.UxTot[i] * self.scale * self.fps / self.paramVecTime['step']

            self.grid_y, self.grid_x = (
                self.grid_y - self.origin[1]) * self.scale, (self.grid_x - self.origin[0]) * self.scale
            self.invertYaxis()
            self.XT = Interpolate.npGrid2TargetPoint2D(
                self.grid_x, self.grid_y)

            self.interp_params = dict(Radius=self.interp_params['Radius'] * self.scale,  
                                      Sharpness=self.interp_params['Sharpness'],
                                      kernel='Gaussian')
            self.filters_params['RadiusF'] = self.filters_params['RadiusF']* self.scale
            self.filters_params['range_Vx'][0] = self.filters_params['range_Vx'][0]* self.scale* self.fps
            self.filters_params['range_Vx'][1] = self.filters_params['range_Vx'][1]* self.scale* self.fps

            self.filters_params['range_Vy'][0] = self.filters_params['range_Vy'][0]* self.scale* self.fps
            self.filters_params['range_Vy'][1] = self.filters_params['range_Vy'][1]* self.scale* self.fps
             
            self.paramPlot = {'vecX': self.vecX,
                              'vecY': self.vecY,
                              'extentFrame': [(self.paramPlot['extentFrame'][0] - self.origin[0]) * self.scale,
                                              (self.paramPlot['extentFrame'][1] - self.origin[0]) * self.scale,
                                              -(self.paramPlot['extentFrame'][2] - self.origin[1]) * self.scale,
                                              -(self.paramPlot['extentFrame'][3] - self.origin[1]) * self.scale,
                                              ],
                              'unit': unit,
                              'vlim': [vlim * self.scale * self.fps / self.paramVecTime['step'] for vlim in self.paramPlot['vlim']]}

            if self.display:
                plt.close(self.num)
                self.opyfDisp = Render.opyfDisplayer(
                    **self.paramPlot, num=self.num)

    def invertYaxis(self):
        self.vecY = -self.vecY
        self.grid_y = -self.grid_y
        if hasattr(self, 'Ux'):
            self.gridVx = - self.gridVx
            self.Uy = -self.Uy
        if hasattr(self, 'X'):
            self.X = self.X * np.array([1, -1])
            self.V = self.V * np.array([1, -1])
            self.Xaccu = self.Xaccu * np.array([1, -1])
            self.Vaccu = self.Vaccu * np.array([1, -1])
            self.Xdata = [(X * np.array([1, -1])) for X in self.Xdata]
            self.Vdata = [(V * np.array([1, -1])) for V in self.Vdata]
        if hasattr(self, 'UyTot'):
            for i in range(len(self.UyTot)):
                self.UyTot[i] = -self.UyTot[i]

    def writeGoodFeaturesPositionsAndDisplacements(self,
            fileFormat='hdf5',
            outFolder='.',
            filename=None,
            fileSequence=False):
        self.filename = filename

        if not self.scaled:
            XpROI = []
            for x in self.Xdata:
                XpROI.append(x[:, 0:2] + self.ROI[0:2])
            XpROI = np.array(XpROI,dtype=object)
        else:
            XpROI = np.copy(self.Xdata)
        if filename is None:
            filename = 'good_features_positions_and_opt_flow_from_frame' + str(self.vec[0]) + 'to' + str(
                self.vec[-1]) + '_with_step_' + str(self.paramVecTime['step']) + '_and_shift_' + str(self.paramVecTime['shift'])
        if fileFormat == 'hdf5':
            Files.hdf5_WriteUnstructured2DTimeserie(
                outFolder + '/' + filename + '.' + fileFormat, self.Time, XpROI, self.Vdata)
        elif fileFormat == 'csv':
            if not fileSequence:
                timeT = np.array(
                    [self.Time, self.Time + self.paramVecTime['step']]).T
                Files.csv_WriteUnstructured2DTimeserie2(
                    outFolder + '/' + filename + '.' + fileFormat, timeT, XpROI, self.Vdata)
            elif fileSequence:
                Files.mkdir2(outFolder + '/' + filename)
                for x, v, t in zip(XpROI, self.Vdata, self.Time):
                    Files.write_csvTrack2D(
                        outFolder + '/' + filename + '/' + format( t,'04.0f') + '_to_' +  format(   t + self.paramVecTime['step'],   '04.0f') + '.' + fileFormat, x, v)

        self.writeImageProcessingParamsJSON(outFolder=outFolder)

    def writeVelocityField(
            self,
            fileFormat='hdf5',
            outFolder='.',
            filename=None,
            fileSequence=False,
            saveParamsImgProc=True):
        # for export in the image referential only if scaling and origin has
        # not been attributed
        self.filename = filename
        if len(self.UxTot) == 0:
            sys.exit( '[Warning] the following method should be run to produce an interpolated field that can be saved {extractGoodFeaturesPositionsDisplacementsAndInterpolate} or {extractGoodFeaturesDisplacementsAccumulateAndInterpolate}')

        vecXpROI = np.copy(self.vecX)
        vecYpROI = np.copy(self.vecY)
        if not self.scaled:
            vecXpROI = self.vecX + self.ROI[0]
            vecYpROI = self.vecY + self.ROI[1]

        self.UxTot = np.array(self.UxTot)
        self.UyTot = np.array(self.UyTot)

        if self.fieldResults == 'time-serie':
            if filename is None:
                self.filename = 'velocity_field_from_frame_' + str(self.vec[0]) + '_to_' + str(self.vec[-1]) + '_with_step_' + str(
                    self.paramVecTime['step']) + '_and_shift_' + str(self.paramVecTime['shift'])
            self.variables = [['Ux_[' + self.unit[0] +'.' +self.unit[1] +
                               '-1]', self.UxTot], ['Uy_[' +  self.unit[0] +'.' +self.unit[1] + '-1]', self.UyTot]]
            if fileFormat == 'hdf5':
                Files.hdf5_Write(outFolder +     '/' +   self.filename +'.' +fileFormat, [['T_[' + self.unit[1] +']', self.Time], ['X_[' +self.unit[0] +']', vecXpROI], ['Y_[' +self.unit[0] +']', vecYpROI]], self.variables)

            if fileFormat == 'tecplot' or fileFormat == 'csv' or fileFormat == 'tec':
                for k in range(len(self.Time)):
                    VT = Interpolate.npGrid2TargetPoint2D(
                        self.UxTot[k], self.UyTot[k])
                    variablesTecplot = [['Ux_[' + self.unit[0] + '.' + self.unit[1] + '^{-1}]', VT[:, 0]], ['Uy_[' + self.unit[0] +  '.' + self.unit[1] +  '-1]', VT[:, 1]]]
                    self.filename = 'velocity_field_from_frame_' + \
                        str(self.vec[2 * k]) + '_to_' + str(self.vec[2 * k + 1])
                    if fileFormat == 'tecplot' or fileFormat == 'tec':
                        Files.tecplot_WriteRectilinearMesh(str(outFolder +'/' + format(k, '04.0f') +'_' + self.filename + '.' + fileFormat), vecXpROI, vecYpROI, variablesTecplot)
                    if fileFormat == 'csv':
                        Files.csv_WriteRectilinearMesh(str(outFolder + '/' +  format( k, '04.0f') + '_' + self.filename + '.' + fileFormat), vecXpROI, vecYpROI, variablesTecplot)

        elif self.fieldResults == 'accumulation':
            VT = Interpolate.npGrid2TargetPoint2D(self.UxTot[0], self.UyTot[0])
            if filename is None:
                #                filename = 'velocity_field_with_accumulation_of_features_velocities_from_frame_' + str(self.vec[0]) + '_to_' + str(self.vec[-1]) + '_with_step_' + str(
                # self.paramVecTime['step']) +
                # '_and_shift_'+str(self.paramVecTime['shift'])
                self.filename = 'frame_' + str(self.vec[0]) + '_to_' + str(self.vec[-1]) + '_with_step_' + str(
                    self.paramVecTime['step']) + '_and_shift_' + str(self.paramVecTime['shift'])
            self.variables = [['Ux_[' + self.unit[0] + '.' + self.unit[1] +
                               '-1]', self.UxTot], ['Uy_[' +  self.unit[0] + '.' + self.unit[1] + '-1]', self.UyTot]]
            if fileFormat == 'hdf5':
                Files.hdf5_Write(outFolder + '/' + self.filename + '.' +
                                 fileFormat, [['T_[' +  self.unit[1] + ']', self.Time], ['X_[' + self.unit[0] + ']', vecXpROI], ['Y_[' +  self.unit[0] +']', vecYpROI]], self.variables)

            if fileFormat == 'tecplot' or fileFormat == 'csv' or fileFormat == 'tec':
                variablesTecplot = [['Ux_[' + self.unit[0] + '.' + self.unit[1] + '^{-1}]', VT[:, 0]], ['Uy_[' +   self.unit[0] +  '.' +  self.unit[1] + '^{-1}]', VT[:, 1]]]

                if fileFormat == 'tecplot' or fileFormat == 'tec':
                    Files.tecplot_WriteRectilinearMesh(
                        outFolder +
                        '/' +
                        self.filename +
                        '.' +
                        fileFormat,
                        vecXpROI,
                        vecYpROI,
                        variablesTecplot)

                if fileFormat == 'csv':
                    Files.csv_WriteRectilinearMesh(
                        outFolder +
                        '/' +
                        self.filename +
                        '.' +
                        fileFormat,
                        vecXpROI,
                        vecYpROI,
                        variablesTecplot)

        if saveParamsImgProc:
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
# TODO export this method in Files

    def writeTracks( self,
            filename=None,
            fileFormat='csv',
            outFolder='.',
            saveParamsImgProc=True):
        import csv
        self.filename = filename
        print('')

        TL = self.tracks_params['track_len']
        B = 0
        if TL < self.paramVecTimeTracks['Ntot']:
            print('Track_length has been set at [' +
                  str(TL) +
                  '] and step at [' +
                  str(self.paramVecTimeTracks['step']) +
                  ']')
            print('Consequently, and considering the tracking processing plan, no tracks before frame {last_frame - Track_length*step}, i.e. ' + str(
                self.vecTracks[-1] - (TL - 1) * self.paramVecTimeTracks['step']) + ' will be saved')
            B = TL
        if filename is None:
            filename = 'tracks_from_frame' + str(self.vecTracks[-B]) + 'to' + str(
                self.vecTracks[-1]) + '_with_step_' + str(self.paramVecTimeTracks['step'])
        f = open(filename, 'w')
        writer = csv.DictWriter(
            f, fieldnames=['track_index', 'frame_index', 'X', 'Y', 'Vx', 'Vy'])
        writer.writeheader()
        N = 0

        for (tr, vtr) in zip(self.tracks, self.vtracks):
            l = len(tr)
            for i in range(l):
                if self.scaled:
                    corr = self.scale * self.fps / self.paramVecTime['step']
                    writer.writerow({'track_index': N,
                                     'frame_index': self.vecTracks[-l + i],
                                     'X': tr[i][0] * self.scale,
                                     'Y': tr[i][1] * self.scale,
                                     'Vx': vtr[i][0] * corr,
                                     'Vy': vtr[i][1] * corr})
                else:
                    writer.writerow({'track_index': N,
                                     'frame_index': self.vecTracks[-l + i],
                                     'X': tr[i][0] + self.ROI[0],
                                     'Y': tr[i][1] + self.ROI[1],
                                     'Vx': vtr[i][0],
                                     'Vy': vtr[i][1]})

            N += 1
        f.close()
        print(
            '[log] Tracks Data saved in ' +
            outFolder +
            '/' +
            filename +
            '.' +
            fileFormat)

        if saveParamsImgProc:
            self.writeImageProcessingParamsJSON(outFolder=outFolder)

    def writeImageProcessingParamsJSON(self, outFolder='.', filename=None):

        fulldict = {
            'lk_params': self.lk_params,
            'feature_params': self.feature_params,
            'filters_params': self.filters_params,
            'interp_parameter': self.interp_params,
            'vecTime_params': self.paramVecTime}
        fulldict['fieldResults'] = self.fieldResults
        Files.writeImageProcessingParamsJSON(
            fulldict, outFolder=outFolder, filename=filename)

    def set_stabilization( self,
            mask=None,
            vlim=[  0, 80],
            mute=True,
            close_at_reset=False):
        if mask is None:
            mask = np.ones((self.Hvis, self.Lvis))
        if self.processingMode == 'video':
            self.stab = opyf.videoAnalyzer(
                self.video_src, mute=mute, display=(
                    mute == False), close_at_reset=close_at_reset, num='stab')
        else:
            self.stab = opyf.frameSequenceAnalyzer(
                self.folder_src, mute=mute, display=(
                    mute == False), close_at_reset=close_at_reset, num='stab')
        self.stab.mask = mask
        self.stab.set_vlim(vlim)
        self.stab.set_vecTime(starting_frame=self.vec[0])
        self.stab.set_goodFeaturesToTrackParams(qualityLevel=0.05)

        self.stabilizeOn = True

    def stabilize(self, s):
        print('-------------Stabilization Start -------------')
        if s - self.vec[0] > 0:
            self.stab.set_vecTime(
                Ntot=1,
                starting_frame=self.vec[0],
                step=s - self.vec[0])
            self.stab.extractGoodFeaturesAndDisplacements()
            transformation_rigid_matrix, rigid_mask = cv2.estimateAffine2D(
                self.stab.X + self.stab.V/2, self.stab.X - self.stab.V/2 ) #because the position of the displacement is half the distance btw GFTs
            dst = cv2.warpAffine(
                self.stab.vis,
                transformation_rigid_matrix,
                (self.stab.Lvis,
                 self.stab.Hvis))
            self.vis = dst
        print('-------------Stabilization End -------------')

    def set_birdEyeViewProcessing(self, image_points,
                                  model_points,
                                  pos_bird_cam,
                                  rotation=np.array([[1, 0, 0.], [0, -1, 0.], [0., 0., -1.]]),
                                  scale=True, framesPerSecond=30, saveOutPut=None,
                                  axisSize=5):

        focal_length = self.Lvis
        center = (self.Lvis / 2, self.Hvis / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )
        _, camera_matrix_inv = cv2.invert(camera_matrix)
        dist_coeffs = np.zeros((4, 1))
        ret, self.rvec1, self.tvec1 = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs)
        self.R1, _ = cv2.Rodrigues(self.rvec1)
        self.RZ = rotation
        self.rvec_z, _ = cv2.Rodrigues(self.RZ)
        self.tvec_z = np.array(
            [-pos_bird_cam[0], -pos_bird_cam[1], -pos_bird_cam[2]], ndmin=2).transpose()
        normal = np.array([0, 0, -1], ndmin=2)
        self.normal1 = np.dot(self.R1, normal.transpose())
        origin = np.array([0, 0, 0], ndmin=2)
        origin1 = np.dot(self.R1, origin.transpose()) + self.tvec1
        self.d_inv1 = 1.0 / np.dot(self.normal1.transpose(), origin1)

        homography_euclidean = Tools.computeHomography(
            self.R1, self.tvec1, self.RZ, self.tvec_z, self.d_inv1, self.normal1)
        homography = camera_matrix @ homography_euclidean @ camera_matrix_inv
        self.homography = homography / homography[2, 2]
        homography_euclidean = homography_euclidean / \
            homography_euclidean[2, 2]
        self.homography = homography
        img_wraped = np.copy(self.vis)
        img_vis = np.copy(self.vis)
        img_vis = cv2.drawFrameAxes(
            img_vis,
            camera_matrix,
            dist_coeffs,
            self.rvec1,
            self.tvec1,
            axisSize)
        img_wraped = cv2.warpPerspective(
            img_wraped, self.homography, (self.Lvis, self.Hvis))
       # self.opyfDisp.plotPointsUnstructured(axisPoints,axisPoints,vis=img_wraped)
        self.figBET, [ax1, ax2] = plt.subplots(
            2, 1, num='Bird Eye Transformation')
        self.figBET.clear()
        [ax1, ax2] = self.figBET.subplots(2, 1)
        self.figBET.suptitle('Bird Eye Transformation')
        self.figBET.set_size_inches(10, 20)
        ax1.imshow(img_vis)
        axisPoints, _ = cv2.projectPoints(
            model_points, self.rvec1, self.tvec1, camera_matrix, (0, 0, 0, 0))

        ax1.scatter(axisPoints[:, 0, 0], axisPoints[:, 0, 1], s=300,
                    linewidths=0.5, marker='P', color=(1, 1, 1, 0.7), zorder=2)
        self.color = []
        props = dict(facecolor=(1, 1, 1), alpha=0.5, pad=2)
        for i in range(len(axisPoints)):
            self.color.append( [np.random.uniform(), np.random.uniform(), np.random.uniform()])
            ax1.scatter(axisPoints[i, 0, 0],
                        axisPoints[i,  0, 1],
                        s=200,
                        linewidths=0.5,
                        marker='+',
                        color=self.color[i],
                        zorder=3)
            ax1.text(axisPoints[i][0,
                                   0],
                     axisPoints[i][0,
                                   1],
                     '        X: ' + format(model_points[i][0],
                                            '2.2f') + ' m Y: ' + format(model_points[i][1],   '2.2f') + ' m Z: ' + format(model_points[i][2], '2.2f') + ' m   ',
                     fontsize=6,
                     bbox=props,
                     zorder=1)
        ax1.set_xlim([0, self.Lvis])
        ax1.set_ylim([self.Hvis, 0])
        # self.opyfDisp.ax.scatter(axisPoints[:,0,0],axisPoints[:,0,1],s=300,linewidths=0.5,marker='P',color=(1,1,1,0.7),zorder=2)

        axisPoints, _ = cv2.projectPoints(
            model_points, self.rvec_z, self.tvec_z, camera_matrix, (0, 0, 0, 0))
        ax2.scatter(axisPoints[:, 0, 0], axisPoints[:, 0, 1], s=300,
                    linewidths=0.5, marker='P', color=(1, 1, 1, 0.7), zorder=2)

        for i in range(len(axisPoints)):
            ax2.scatter(axisPoints[i,  0,  0],
                        axisPoints[i,  0,  1],
                        s=200,
                        linewidths=0.5,
                        marker='+',
                        color=self.color[i],
                        zorder=3)
            # ax2.text(axisPoints[i][0,0],
            #   axisPoints[i][0,1],'        X: '+format(model_points[i][0],'2.2f')+' m Y: '+format(model_points[i][1],'2.2f')+' m Z: '+format(model_points[i][2],'2.2f')+' m   ',
            #   fontsize=6, bbox=props,zorder=1)

        img_wraped = cv2.drawFrameAxes(
            img_wraped,
            camera_matrix,
            dist_coeffs,
            self.rvec_z,
            self.tvec_z,
            axisSize)
        ax2.imshow(img_wraped)
        ax2.set_xlim([0, self.Lvis])
        ax2.set_ylim([self.Hvis, 0])  # invert axis to obtain the correct view
        if plt.rcParams['backend'] in mpl.rcsetup.interactive_bk:
            self.figBET.show()
        time.sleep(0.1)

        if scale:
            points = np.array([(0, 0, 0), (1., 0, 0)])
            axisPoints, _ = cv2.projectPoints(
                points, self.rvec_z, self.tvec_z, camera_matrix, (0, 0, 0, 0))
            pxPerM = ((axisPoints[0, 0, 0] - axisPoints[1, 0, 0]) **
                      2 + (axisPoints[0, 0, 1] - axisPoints[1, 0, 1])**2)**0.5

            self.scaleData(metersPerPx=1 / pxPerM, framesPerSecond=framesPerSecond,
                           origin=[axisPoints[0, 0, 0], axisPoints[0, 0, 1]])
        self.birdEyeMod = True

        if saveOutPut is not None:
            self.figBET.savefig(saveOutPut)

    def transformBirdEye(self):
        self.vis = cv2.warpPerspective(
            self.vis, self.homography, (self.Lvis, self.Hvis))


# #    plt.subplot(121),plt.imshow(img),plt.title('Input')
# #    plt.subplot(122),plt.imshow(dst),plt.title('Output')
# #    plt.show()
#     return img_wraped

#            for x,v in zip(Xdata,Vdata):

    # def extractGoodFeaturesPositionsAndExtractFeatures(self, display=False,
    # windowSize=(16, 16)):

    #     for pr, i in zip(self.prev, self.vec):

    #         #    frame=frame[:,:,1]
    #         l = self.listD[i]

    # vis = cv2.imread(self.folder_src+'/'+l)  # special for tiff images

    #         vis = Render.CLAHEbrightness(
    #             vis, 0, tileGridSize=(20, 20), clipLimit=2)
    #         if len(np.shape(vis)) == 3:
    #             current_gray = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)
    #         else:
    #             current_gray = vis/media/gauthier/Data-Gauthier/Gauthier/Download/MNT_1m_Brague@Biot.tif
    #         current_gray = current_gray[self.ROI[1]:(
    # self.ROI[3]+self.ROI[1]), self.ROI[0]:(self.ROI[2]+self.ROI[0])]

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
        processingMode = 'video'
        self.video_src = video_src
        cap = cv2.VideoCapture(self.video_src)
        self.ret, frameInit = cap.read()
        self.sizeFrameInMByte = sys.getsizeof(frameInit) / 1e6
        if not self.ret:
            print('Error: the video file path might be wrong')
            # sys.exit()
        frameInit = Tools.convertToGrayScale(frameInit)
        number_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        super().__init__(
            frameInit=frameInit,
            number_of_frames=number_of_frames,
            processingMode=processingMode,
            cap=cap,
            **args)
        # Analyzer.__init__(self, **args)


class frameSequenceAnalyzer(Analyzer):
    def __init__(self, folder_src, **args):
        processingMode = 'image sequence'
        listD = os.listdir(folder_src)
        listD = sorted(listD, key=self.natural_keys)
        number_of_frames = len(listD)
        frameInit = cv2.imread(folder_src + '/' + listD[0])
        frameInit = Tools.convertToGrayScale(frameInit)
        super().__init__(
            frameInit=frameInit,
            number_of_frames=number_of_frames,
            processingMode=processingMode,
            listD=listD,
            folder_src=folder_src,
            **args)
        # Analyzer.__init__(self, **args)

    def atoi(self, text):
        return int(text) if text.isdigit() else text

    def natural_keys(self, text):
        '''
        alist.sort(key=natural_keys) sorts in human order
        http://nedbatchelder.com/blog/200712/human_sorting.html
        (See Toothy's implementation in the comments)
        '''
        return [self.atoi(c) for c in re.split(r'(\d+)', text)]


# TODO    def writeGoodFeaturesPositionsAndExtraction(self,fileFormat='hdf5',imgFormat='png',outFolder='',filename=None,fileSequence=False):
#        if filename is None:
#            filename='Goof_features_positions_from_frame'+ str(self.vec[0]) + 'to' + str(self.vec[-2]) + '_with_shift_'+str(self.paramVecTime['shift'])
#
#        for samples in self.samplesMat:
#           folderFrame=outFolder+'/'+filename+'/'+format(t,'04.0f')+'_to_'+format(t+self.paramVecTime['step']
#           Files.mkdir2()
#           cv2.imwrite()

# %%
