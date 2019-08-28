#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 08:59:41 2019

@author: gauthier ROUSSEAU
"""

import os
import sys
import numpy as np
import cv2
from opyf import MeshesAndTime, Track, Render, Files, Tools, Interpolate, Filters
import matplotlib.pyplot as plt


class Analyzer():
    def __init__(self, **args):
        self.scaled = False
        self.ROI = args.get(
            'imageROI', [0, 0, self.frameInit.shape[1], self.frameInit.shape[0]])
        self.cropFrameInit = self.frameInit[self.ROI[1]:(
            self.ROI[3]+self.ROI[1]), self.ROI[0]:(self.ROI[2]+self.ROI[0])]
        self.Hvis, self.Lvis = self.cropFrameInit.shape
        self.mask = np.ones(self.cropFrameInit.shape)
        self.frameAv = np.zeros(self.cropFrameInit.shape)

        self.set_goodFeaturesToTrackParams(**args)
        self.set_opticalFlowParams(**args)
        self.set_filtersParams(**args)
        self.set_interpolationParams(**args)
        self.vlimPx = args.get('vlim', [-np.inf, np.inf])

        self.reset()

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
                          'num': args.get('num', 'opyfPlot'),
                          'grid': args.get('grid', True),
                          'vlim': args.get('vlim', [0, 40])}
        self.setGridToInterpolateOn(0, self.Lvis, 4, 0, self.Hvis, 4)
        self.gridMask = np.ones((self.Hgrid, self.Lgrid))

        self.setVecTime(**args)
        print('number of frames : ' + str(self.number_of_frames))

    def set_goodFeaturesToTrackParams(self, **args):

        self.feature_params = dict(maxCorners=args.get('maxCorners', 40000),
                                   qualityLevel=args.get(
                                       'qualityLevel', 0.005),
                                   minDistance=args.get('minDistance', 5),
                                   blockSize=args.get('blockSize', 16))

    def set_opticalFlowParams(self, **args):

        self.lk_params = dict(winSize=args.get('winSize', (16, 16)),
                              maxLevel=args.get('maxLevel', 3),
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.03))

    def set_filtersParams(self, **args):
        self.filters_params = dict(RadiusF=args.get('RadiusF', 30),
                                   minNperRadius=args.get('minNperRadius', 0),
                                   maxDevinRadius=args.get(
                                       'maxDevinRadius', np.inf),
                                   DGF=args.get('wayBackGoodFlag', np.inf))

    def set_interpolationParams(self, **args):
        self.interp_params = dict(Radius=args.get('Radius', 30),  # it is not necessary to perform unterpolation on a high radius since we have a high number of values
                                  Sharpness=args.get(' Sharpness', 8),
                                  kernel=args.get(' kernel', 'Gaussian'))

    def set_vlim(self, vlim):
        if self.scaled == False:
            self.vlimPx = vlim
        else:
            self.vlimPx = [
                v/(self.scale*self.fps/self.paramVecTime['step']) for v in vlim]

        self.paramPlot['vlim'] = vlim
        self.opyfDisp = Render.opyfDisplayer(**self.paramPlot)

    def setGridToInterpolateOn(self, pixLeft=0, pixRight=0, stepHor=2, pixUp=0, pixDown=0, stepVert=2, ROI=None, stepGrid=None):
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
        self.grid_y, self.grid_x, self.gridVx, self.gridVy,  self.Hgrid, self.Lgrid = MeshesAndTime.setGridToInterpolateOn(
            pixLeft, pixRight, stepHor, pixUp, pixDown, stepVert)
        if self.scaled == True:
            self.grid_y, self.grid_x = - \
                (self.grid_y-self.origin[1]) * \
                self.scale, (self.grid_x-self.origin[0])*self.scale
        self.vecX = self.grid_x[0, :]
        self.vecY = self.grid_y[:, 0]
        self.XT = Interpolate.npGrid2TargetPoint2D(self.grid_x, self.grid_y)
        self.XT = Interpolate.npGrid2TargetPoint2D(self.grid_x, self.grid_y)
        self.paramPlot['vecX'] = self.vecX
        self.paramPlot['vecY'] = self.vecY
        self.opyfDisp = Render.opyfDisplayer(**self.paramPlot)
        self.Ux, self.Uy = np.zeros((self.Hgrid, self.Lgrid)), np.zeros(
            (self.Hgrid, self.Lgrid))
        self.gridMask = np.ones((self.Hgrid, self.Lgrid))

    def set_imageParams(self, **args):
        self.rangeOfPixels = args.get('rangeOfPixels', [0, 255])
        self.imreadOption = args.get('imreadOption', 0)
        if self.imreadOption == 'ANYDEPTH':
            self.imreadOption = 2

    def setVecTime(self, **args):
        self.paramVecTime = {'starting_frame': args.get('starting_frame', 0),
                             'step': args.get('step', 1),
                             'shift': args.get('shift', 1),
                             'Ntot': args.get('Ntot', 1)}

        self.vec, self.prev = MeshesAndTime.setVecTime(framedeb=self.paramVecTime['starting_frame'],
                                                       step=self.paramVecTime['step'],
                                                       shift=self.paramVecTime['shift'],
                                                       Ntot=self.paramVecTime['Ntot'])
        if self.vec[-1] > self.number_of_frames:
            print('----- Error ----')
            print('Your processing plan is not compatible with the frame set')
            print(
                'Consider that [starting_frame+step+(Ntot-1)*shift]  must be smaller than the number of frames')
            sys.exit()
        print('[The following image processing plan has been set]')
        if self.processingMode == 'image sequence':
            for pr, i in zip(self.prev, self.vec):
                if pr == False:
                    print(
                        '--> detect Good Features to Track on image [' + self.listD[i]+']')
                    file_prev = self.listD[i]
                else:
                    print(
                        '--> measure diplacements between images [' + file_prev + '] and [' + self.listD[i] + ']')

        if self.processingMode == 'video':
            for pr, i in zip(self.prev, self.vec):
                if pr == False:
                    print(
                        '--> detect Good Features to Track on frame [' + str(i)+']')
                    file_prev = str(i)
                else:
                    print(
                        '--> measure diplacements between frame [' + file_prev + '] and [' + str(i) + ']')
        self.Time = self.vec[0:-1:2]
        # initilize self.vis with the first frame of the set
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

#TODO and test
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
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, self.vis = self.cap.read()
        self.vis = self.vis[self.ROI[1]:(
            self.ROI[3]+self.ROI[1]), self.ROI[0]:(self.ROI[2]+self.ROI[0])]

    def stepGoodFeaturesToTrackandOpticalFlow(self, pr, i):
        if pr == False:
            self.prev_gray = None

        self.readFrame(i)

        gray = Tools.convertToGrayScale(self.vis)

        if self.rangeOfPixels != [0, 255] or self.frameAv is not None:
            gray = np.array(gray, dtype='float32')
            gray = (
                (gray-self.rangeOfPixels[0])/(self.rangeOfPixels[1]-self.rangeOfPixels[0]))*255-self.frameAv
            gray[np.where(gray < 0)] = 0
            gray[np.where(gray > 255)] = 255
            gray = np.array(gray, dtype='uint8')

        current_gray = Render.CLAHEbrightness(
            gray, 0, tileGridSize=(20, 20), clipLimit=2)
#        self.vis=Render.CLAHEbrightness(self.vis,0,tileGridSize=(20,20),clipLimit=2)

        self.prev_gray, self.X, self.V = Track.opyfFlowGoodFlag(current_gray,
                                                                self.prev_gray,
                                                                self.feature_params,
                                                                self.lk_params,
                                                                ROI=self.ROImeasure,
                                                                vmin=self.vlimPx[0],
                                                                vmax=self.vlimPx[1],
                                                                mask=self.mask,
                                                                DGF=self.filters_params['DGF'])

        if pr == True:

            # filters ###### important since we have a low quality level for the Good Feature to track and a high Distance Good Flag
            if self.filters_params['minNperRadius'] > 0:
                Dev, Npoints, stD = Filters.opyfFindPointsWithinRadiusandDeviation(
                    self.X, (self.V[:, 0]**2+self.V[:, 1]**2), self.filters_params['RadiusF'])
                self.X = Filters.opyfDeletePointCriterion(
                    self.X, Npoints, climmin=self.filters_params['minNperRadius'])
                self.V = Filters.opyfDeletePointCriterion(
                    self.V, Npoints, climmin=self.filters_params['minNperRadius'])
            if self.filters_params['maxDevinRadius'] != np.inf:
                Dev, Npoints, stD = Filters.opyfFindPointsWithinRadiusandDeviation(
                    self.X, self.V[:, 1], self.filters_params['RadiusF'])
                self.X = Filters.opyfDeletePointCriterion(
                    self.X, Dev, climmax=self.filters_params['maxDevinRadius'])
                self.V = Filters.opyfDeletePointCriterion(
                    self.V, Dev, climmax=self.filters_params['maxDevinRadius'])
                Dev, Npoints, stD = Filters.opyfFindPointsWithinRadiusandDeviation(
                    self.X, self.V[:, 0], self.filters_params['RadiusF'])
                self.X = Filters.opyfDeletePointCriterion(
                    self.X, Dev, climmax=self.filters_params['maxDevinRadius'])
                self.V = Filters.opyfDeletePointCriterion(
                    self.V, Dev, climmax=self.filters_params['maxDevinRadius'])

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
            print('Displacement max = '+str(np.max(self.V)) +
                  ' '+self.unit[0]+'/'+self.unit[1])

    def reset(self):

        self.Xdata = []
        self.Vdata = []
        self.incr = 0
        self.UxTot = []
        self.UyTot = []

    def extractGoodFeaturesPositionsAndDisplacements(self, display=False, saveImgPath=None, imgFormat='.png', **args):
        self.reset()

        for pr, i in zip(self.prev, self.vec):
            self.stepGoodFeaturesToTrackandOpticalFlow(pr, i)
            if len(self.X) > 0:

                self.Xdata.append(self.X)
                self.Vdata.append(self.V)

                self.showXV(self.X, self.V, vis=self.vis,
                            display=display, **args)
                if saveImgPath is not None:
                    self.opyfDisp.fig.savefig(saveImgPath+'/'+display+'_'+format(
                        i, '04.0f')+'_to_'+format(i+self.paramVecTime['step'], '04.0f')+'.'+imgFormat)
                plt.pause(0.01)

    def interpolateOnGrid(self, mode='sequence'):

        self.interpolatedVelocities = Interpolate.npInterpolateVTK2D(
            self.X, self.V, self.XT, ParametreInterpolatorVTK=self.interp_params)

        self.gridMask[np.where(self.gridMask == 0)] = np.nan
        self.Ux = Interpolate.npTargetPoints2Grid2D(
            self.interpolatedVelocities[:, 0], self.Lgrid, self.Hgrid)*self.gridMask
        self.Uy = Interpolate.npTargetPoints2Grid2D(
            self.interpolatedVelocities[:, 1], self.Lgrid, self.Hgrid)*self.gridMask

    def extractGoodFeaturesPositionsDisplacementsAndInterpolate(self, display=False, saveImgPath=None, Type='norme', imgFormat='png', **args):
        self.reset()
        for pr, i in zip(self.prev, self.vec):
            self.stepGoodFeaturesToTrackandOpticalFlow(pr, i)
            if len(self.X) > 0:
                self.Xdata.append(self.X)
                self.Vdata.append(self.V)
                self.interpolateOnGrid()
                self.UxTot.append(np.reshape(
                    self.interpolatedVelocities[:, 0], (self.Hgrid, self.Lgrid)))
                self.UyTot.append(np.reshape(
                    self.interpolatedVelocities[:, 1], (self.Hgrid, self.Lgrid)))
                Field = Render.setField(self.Ux, self.Uy, Type)
                if display == 'field':
                    self.opyfDisp.plotField(Field, vis=self.vis, **args)
                    if saveImgPath is not None:
                        self.opyfDisp.fig.savefig(saveImgPath+'/'+display+'_'+format(
                            i, '04.0f')+'_to_'+format(i+self.paramVecTime['step'], '04.0f')+'.'+imgFormat)
                    plt.pause(0.01)

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
            self.Time = self.Time/self.fps*self.paramVecTime['step']

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

            self.opyfDisp = Render.opyfDisplayer(**self.paramPlot)

    def invertYaxis(self):
        self.vecY = -self.vecY
        self.grid_y = -self.grid_y
        self.gridVx = -self.gridVx
        self.Uy = -self.Uy
        self.Xdata = [(X*np.array([1, -1])) for X in self.Xdata]
        self.Vdata = [(V*np.array([1, -1])) for V in self.Vdata]

    def writeGoodFeaturesPositionsAndDisplacements(self, fileFormat='hdf5', outFolder='.', filename=None, fileSequence=False):
        if filename is None:
            filename = 'good_features_positions_and_opt_flow_from_frame' + str(self.vec[0]) + 'to' + str(
                self.vec[-1]) + '_with_step_' + str(self.paramVecTime['step']) + '_and_shift_'+str(self.paramVecTime['shift'])
        if fileFormat == 'hdf5':
            Files.hdf5_WriteUnstructured2DTimeserie(
                outFolder+'/'+filename+'.'+fileFormat, self.Time, self.Xdata, self.Vdata)
        elif fileFormat == 'csv':
            if fileSequence == False:
                Files.csv_WriteUnstructured2DTimeserie(
                    outFolder+'/'+filename+'.'+fileFormat, self.Time, self.Xdata, self.Vdata)
            elif fileSequence == True:
                Files.mkdir2(outFolder+'/'+filename)
                for x, v, t in zip(self.Xdata, self.Vdata, self.Time):
                    Files.write_csvTrack2D(outFolder+'/'+filename+'/'+format(t, '04.0f')+'_to_'+format(
                        t+self.paramVecTime['step'], '04.0f')+'.'+fileFormat, x, v)

        self.writeImageProcessingParamsJSON(outFolder=outFolder)

    def writeVelocityField(self, fileFormat='hdf5', outFolder='.', filename=None, fileSequence=False, saveParamsImgProc=True):
        self.UxTot = np.array(self.UxTot)
        self.UyTot = np.array(self.UyTot)
        if filename is None:
            filename = 'velocity_field_from_frame' + str(self.vec[0]) + 'to' + str(self.vec[-1]) + '_with_step_' + str(
                self.paramVecTime['step']) + '_and_shift_'+str(self.paramVecTime['shift'])
        if fileFormat == 'hdf5':
            variables = [['Ux_['+self.unit[0]+'.'+self.unit[1]+'-1]', self.UxTot],
                         ['Uy_['+self.unit[0]+'.'+self.unit[1]+'-1]', self.UyTot]]
            Files.hdf5_Write(outFolder+'/'+filename+'.'+fileFormat, [['T_['+self.unit[0]+']', self.Time], [
                             'X_['+self.unit[0]+']', self.vecX], ['Y_['+self.unit[0]+']', self.vecY]], variables)
        if saveParamsImgProc == True:
            self.writeImageProcessingParamsJSON(outFolder=outFolder)
 # TODO for csv format
#        elif fileFormat=='csv':
#            if fileSequence==False:
#                Files.csv_WriteUnstructured2DTimeserie(outFolder+'/'+filename+'.'+fileFormat,self.Time,self.Xdata,self.Vdata)
#            elif fileSequence==True:
#                Files.mkdir2(outFolder+'/'+filename)
#                for x,v,t in zip(self.Xdata,self.Vdata,self.Time):
#                    Files.write_csvTrack2D(outFolder+'/'+filename+'/'+format(t,'04.0f')+'_to_'+format(t+self.paramVecTime['step'],'04.0f')+'.'+fileFormat,x,v)
#
#

    def writeImageProcessingParamsJSON(self, outFolder='.', filename=None):

        fulldict = {'lk_params': self.lk_params, 'feature_params': self.feature_params,
                    'filters_params': self.filters_params, 'interp_parameter': self.interp_params}

        Files.writeImageProcessingParamsJSON(
            fulldict, outFolder=outFolder, filename=None)


#            for x,v in zip(Xdata,Vdata):
    def extractGoodFeaturesPositionsAndExtractFeatures(self, display=False, windowSize=(16, 16)):

        for pr, i in zip(self.prev, self.vec):

            #    frame=frame[:,:,1]
            l = self.listD[i]

            vis = cv2.imread(self.folder_src+'/'+l)  # special for tiff images

            vis = Render.CLAHEbrightness(
                vis, 0, tileGridSize=(20, 20), clipLimit=2)
            if len(np.shape(vis)) == 3:
                current_gray = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)
            else:
                current_gray = vis
            current_gray = current_gray[self.ROI[1]:(
                self.ROI[3]+self.ROI[1]), self.ROI[0]:(self.ROI[2]+self.ROI[0])]

            p0 = cv2.goodFeaturesToTrack(current_gray, **self.feature_params)
            X = p0.reshape(-1, 2)
            samples = []
            for x in X:
                featureExtraction = current_gray[int(x[0]-windowSize[0]/2):int(
                    x[0]+windowSize[0]/2), int(x[1]-windowSize[1]/2):int(x[1]+windowSize[1]/2)]
                samples.append(featureExtraction)

            self.samplesMat.append(samples)
            self.Xsamples.append(X)
            if display == 'points':
                self.show(self, X, X, vis, display='points')


class videoAnalyser(Analyzer):
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
    def __init__(self, **args):
        self.processingMode = 'image sequence'
        self.folder_src = args.get('folder_src', './')
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
