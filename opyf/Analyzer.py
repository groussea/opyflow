#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 08:59:41 2019

@author: gauthier ROUSSEAU
"""

import os,sys
import numpy as np
import cv2
from opyf import MeshesAndTime
from opyf import Track
from opyf import Render
from opyf import Files
from opyf import Tools
from opyf import Interpolate
import matplotlib.pyplot as plt
import time




class Analyzer():
    def __init__(self,**args):
#        plt.close('all')
        self.Hvis,self.Lvis=self.frameInit.shape
        self.ROI=[0,0,self.frameInit.shape[1],self.frameInit.shape[0]]
        self.mask=np.ones(self.frameInit.shape) 
        self.frameAv=np.zeros(self.frameInit.shape) 
        self.feature_params = dict( maxCorners = 40000,
                           qualityLevel = 0.005,
                           minDistance = 3,
                           blockSize = 16)
    
    
        self.lk_params = dict( winSize  = (16, 16),
                          maxLevel = 3,
                          criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.03))
        
        self.filters_params = dict(vmin=None, 
                              vmax=None,
                              RadiusF=20.,
                              minNperRadius=2.,
                              maxDevinRadius='nean',
                              DGF=10)
        
        self.interp_params= dict (Radius=20, # it is not necessary to perform unterpolation on a high radius since we have a high number of values
                     Sharpness=8,
                     kernel='Gaussian')

        self.Xdata=[]
        self.Vdata=[]
        self.samplesMat=[]
        self.Xsamples=[]
        self.set_Displayer(**args) 
        self.unit=args.get('unit',['px','deltaT'])                      
        self.set_imageParams(**args) 
        self.setGridToInterpolateOn(0,self.Lvis,4,0,self.Hvis,4)

        self.vecY=args.get('vecY',np.arange(self.Hvis))
        self.setVecTime(**args)
        print('number of frames : '+ str(self.number_of_frames) )


    def setGridToInterpolateOn(self,pixLeft,pixRight,stepHor,pixUp,pixDown,stepVert):
        
        self.grid_y, self.grid_x,self.gridVx, self.gridVy,  self.Hgrid, self.Lgrid=MeshesAndTime.setGridToInterpolateOn(pixLeft,pixRight,stepHor,pixUp,pixDown,stepVert)
        self.vecX=self.grid_x[0,:]
        self.vecY=self.grid_y[:,0]
        self.XT=Interpolate.npGrid2TargetPoint2D(self.grid_x,self.grid_y)
        
    def set_imageParams(self,**args):
        
        self.rangeOfPixels=args.get('rangeOfPixels',[0,255])
        self.imreadOption=args.get('imreadOption',0)
        if self.imreadOption=='ANYDEPTH':
            self.imreadOption=2
            
    def set_Displayer(self,**args):
        
        self.paramPlot={'ScaleVectors':args.get('ScaleVectors',0.1),
             'vecX':args.get('vecX',np.arange(self.Lvis)),
             'vecY':args.get('vecY',np.arange(self.Hvis)),
             'extentFrame':args.get('extentFrame',[0,self.Lvis,self.Hvis,0]), 
             'unit':args.get('unit',['px','deltaT']),
             'Hfig':args.get('Hfig',8),
             'num':args.get('num','opyfPlot'),
             'grid':args.get('grid',True),
             'vlim':args.get('vlim',[0,50])} 
        
        self.opyfDisp=Render.opyfDisplayer(**self.paramPlot) 


    def setVecTime(self,**args):
            self.paramVecTime={'starting_frame':args.get('starting_frame',0),
                                'step':args.get('step',1),
                                'shift':args.get('shift',1),
                                'Ntot':args.get('Ntot',1)} 
             
    
            self.vec,self.prev=MeshesAndTime.setVecTime(framedeb=self.paramVecTime['starting_frame'],
                                                         step=self.paramVecTime['step'],
                                                         shift=self.paramVecTime['shift'],
                                                         Ntot=self.paramVecTime['Ntot'])
            if self.vec[-1]>self.number_of_frames:
                print('----- Error ----')
                print('Your processing plan is not compatible with the frame set')
                print('Consider that [starting_frame+(Ntot-1)*shift+step]  must be smaller than the number_of_frame' )
                sys.exit()
            print('[The following image processing plan has been set]')
            if self.processingMode=='image sequence':
                for pr,i in zip(self.prev,self.vec):
                    if pr==False:
                        print('--> detect Good Features to Track on image [' +self.listD[i]+']')
                        file_prev=self.listD[i]
                    else:
                        print('--> measure diplacements between images [' +file_prev +'] and [' +self.listD[i] +']')
                        
            if self.processingMode=='video':
                for pr,i in zip(self.prev,self.vec):
                    if pr==False:
                        print('--> detect Good Features to Track on frame [' +str(i)+']')
                        file_prev=str(i)
                    else:
                        print('--> measure diplacements between frame [' +file_prev +'] and [' +str(i) +']')             
            self.Time=self.vec[0:-1:2]    

    def initializeAveragedFrameFromFile(self,file,imreadOption=1):
        frameav=cv2.imread(file,imreadOption)  
        frameav=Tools.convertToGrayScale(frameav)
        if frameav is None:
            print('Invalid path for the averaged frame')
        elif frameav.shape!=(self.Hvis,self.Lvis):
            print('Invalid dimensions for the averaged frame')
        else:
            self.frameAv=frameav
        
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
    def runGFTandDisp(self,pr,i):

        if pr==False:
            self.prev_gray=None
        #    frame=frame[:,:,1]
        if self.processingMode=='image sequence':
            l=self.listD[i]
            self.vis=cv2.imread(self.folder_src+'/'+l,self.imreadOption) #2 for tiff images
        elif self.processingMode=='video':
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i) 
            ret, self.vis = self.cap.read()
        
        gray=Tools.convertToGrayScale(self.vis)
        
        if self.rangeOfPixels!=[0,255] or self.frameAv is not None:
            gray=np.array(gray,dtype='float32')
            gray=((gray-self.rangeOfPixels[0])/(self.rangeOfPixels[1]-self.rangeOfPixels[0]))*255-self.frameAv
            gray[np.where(gray<0)]=0
            gray[np.where(gray>255)]=255
            gray=np.array(gray,dtype='uint8') 
   
        gray=Render.CLAHEbrightness(gray,0,tileGridSize=(20,20),clipLimit=2)
#        self.vis=Render.CLAHEbrightness(self.vis,0,tileGridSize=(20,20),clipLimit=2)

        current_gray=gray[self.ROI[1]:(self.ROI[3]+self.ROI[1]),self.ROI[0]:(self.ROI[2]+self.ROI[0])] 
        
        
                  
        self.prev_gray,self.X,self.V=Track.opyfFlowGoodFlag(current_gray,
                                             self.prev_gray,
                                             self.feature_params,
                                             self.lk_params,
                                             ROI=self.ROI,
                                             vmax=self.filters_params['vmax'],
                                             vmin=self.filters_params['vmin'],
                                             mask=self.mask,
                                             DGF=self.filters_params['DGF'])   

        
        if len(self.X)>0:
            print('')
            if self.processingMode=='image sequence':
                print('------- From frame ['+ self.listD[i-self.paramVecTime['step']] +'] to frame [' + self.listD[i] +'] -------')
            if self.processingMode=='video':
                print('------- From frame ['+ str(i-self.paramVecTime['step']) +'] to frame [' + str(i) +'] -------')
            print('Number of Good Feature To Track = '+str(len(self.X)))
            print('Displacement max = '+str(np.max(self.V)))


            
        

    def extractGoodFeaturesPositionsAndDisplacements(self,display=False,saveImgPath=None,imgFormat='.png',**args):
        self.Xdata=[]
        self.Vdata=[]     
        for pr,i in zip(self.prev,self.vec):
            self.runGFTandDisp(pr,i)
            if len(self.X)>0:
                self.Xdata.append(self.X)
                self.Vdata.append(self.V)
                self.showXV(self.X,self.V,self.vis,display=display,**args)
                if saveImgPath is not None:
                    self.opyfDisp.fig.savefig(saveImgPath+'/'+display+'_'+format(i,'04.0f')+'_to_'+format(i+self.paramVecTime['step'],'04.0f')+'.'+imgFormat)
                plt.pause(0.01)

    def interpolateOnGrid(self,mode='sequence'):
        
        self.interpolatedVelocities=Interpolate.npInterpolateVTK2D(self.X,self.V,self.XT,ParametreInterpolatorVTK=self.interp_params)
        NANmask=np.ones((self.Hgrid,self.Lgrid))
        NANmask[np.where(self.mask==0)]=np.nan
        self.Ux=Interpolate.npTargetPoints2Grid2D(self.interpolatedVelocities[:,0],self.Lgrid,self.Hgrid)*NANmask
        self.Uy=Interpolate.npTargetPoints2Grid2D(self.interpolatedVelocities[:,1],self.Lgrid,self.Hgrid)*NANmask
 
    
    def  extractGoodFeaturesPositionsDisplacementsAndInterpolate(self,display=False,saveImgPath=None,Type='norme',imgFormat='.png',**args):
        self.Xdata=[]
        self.Vdata=[]  
        self.interpolatedVelocitiesTotal=np.empty([0,2])
        for pr,i in zip(self.prev,self.vec):
            self.runGFTandDisp(pr,i)
            if len(self.X)>0:
                self.Xdata.append(self.X)
                self.Vdata.append(self.V)
                self.interpolateOnGrid()
                self.interpolatedVelocitiesTotal=np.append(self.interpolatedVelocitiesTotal,self.interpolatedVelocities[:,0:2],axis=0)
                Field=Render.setField(self.Ux,self.Uy,Type)
                if display=='field':
                    self.opyfDisp.plotField(Field,vis=self.vis,**args)
                    if saveImgPath is not None:
                        self.opyfDisp.fig.savefig(saveImgPath+'/'+display+'_'+format(i,'04.0f')+'_to_'+format(i+self.paramVecTime['step'],'04.0f')+'.'+imgFormat)
                    plt.pause(0.01)

            

    def showXV(self,X,V,vis,display='quiver',displayColor=False,**args):
            if display=='quiver':
                             
                self.opyfDisp.plotQuiverUnstructured(X,V,vis=vis,displayColor=displayColor,**args)
                plt.pause(0.01)
                
            if display=='points':
                          
                self.opyfDisp.plotPointsUnstructured(Xdata=X,Vdata=V,vis=vis,displayColor=displayColor,**args)
                plt.pause(0.1)
            
                


    def writeGoodFeaturesPositionsAndDisplacements(self,fileFormat='hdf5',outFolder='.',filename=None,fileSequence=False):
        if filename is None:
            filename='good_features_positions_and_displacements_from_frame'+ str(self.vec[0]) + 'to' + str(self.vec[-1]) + '_with_step_' +str(self.paramVecTime['step']) + '_and_shift_'+str(self.paramVecTime['shift'])
        if fileFormat=='hdf5':
            Files.hdf5_WriteUnstructured2DTimeserie(outFolder+'/'+filename+'.'+fileFormat,self.Time,self.Xdata,self.Vdata)
        elif fileFormat=='csv':
            if fileSequence==False:
                Files.csv_WriteUnstructured2DTimeserie(outFolder+'/'+filename+'.'+fileFormat,self.Time,self.Xdata,self.Vdata)
            elif fileSequence==True:    
                Files.mkdir2(outFolder+'/'+filename)
                for x,v,t in zip(self.Xdata,self.Vdata,self.Time):
                    Files.write_csvTrack2D(outFolder+'/'+filename+'/'+format(t,'04.0f')+'_to_'+format(t+self.paramVecTime['step'],'04.0f')+'.'+fileFormat,x,v)

        self.writeImageProcessingParamsJSON(outFolder=outFolder)
        
    def writeVelocityField(self,fileFormat='hdf5',outFolder='.',filename=None,fileSequence=False,saveParamsImgProc=True):
        self.UxTot=np.reshape(self.interpolatedVelocitiesTotal[:,0],(int(self.paramVecTime['Ntot']),self.Hgrid,self.Lgrid))
        self.UyTot=np.reshape(self.interpolatedVelocitiesTotal[:,1],(int(self.paramVecTime['Ntot']),self.Hgrid,self.Lgrid))
        if filename is None:
            filename='velocity_field_from_frame'+ str(self.vec[0]) + 'to' + str(self.vec[-1]) + '_with_step_' +str(self.paramVecTime['step']) + '_and_shift_'+str(self.paramVecTime['shift'])
        if fileFormat=='hdf5':
            variables=[['Ux_['+self.unit[0]+'.'+self.unit[1]+'-1]',self.UxTot],['Uy_['+self.unit[0]+'.'+self.unit[1]+'-1]',self.UyTot]]
            Files.hdf5_Write(outFolder+'/'+filename+'.'+fileFormat,[['T_['+self.unit[0]+']',self.Time],['X_['+self.unit[0]+']',self.vecX],['Y_['+self.unit[0]+']',self.vecY]],variables)
        if saveParamsImgProc==True:
            self.writeImageProcessingParamsJSON(outFolder=outFolder)
        

    def writeImageProcessingParamsJSON(self,outFolder='.',filename=None):

        fulldict={'lk_params':self.lk_params,'feature_params':self.feature_params,
                  'filters_params':self.filters_params,'interp_parameter':self.interp_params}
        
        Files.writeImageProcessingParamsJSON(fulldict,outFolder=outFolder,filename=None)
        
        
#TODO for csv format
#        elif fileFormat=='csv':
#            if fileSequence==False:
#                Files.csv_WriteUnstructured2DTimeserie(outFolder+'/'+filename+'.'+fileFormat,self.Time,self.Xdata,self.Vdata)
#            elif fileSequence==True:    
#                Files.mkdir2(outFolder+'/'+filename)
#                for x,v,t in zip(self.Xdata,self.Vdata,self.Time):
#                    Files.write_csvTrack2D(outFolder+'/'+filename+'/'+format(t,'04.0f')+'_to_'+format(t+self.paramVecTime['step'],'04.0f')+'.'+fileFormat,x,v)
#                                
#                
                
#            for x,v in zip(Xdata,Vdata):
    def extractGoodFeaturesPositionsAndExtractFeatures(self,display=False,windowSize=(16,16)): 

        for pr,i in zip(self.prev,self.vec):

            #    frame=frame[:,:,1]
            l=self.listD[i]

            vis=cv2.imread(self.folder_src+'/'+l) #special for tiff images

   
            vis=Render.CLAHEbrightness(vis,0,tileGridSize=(20,20),clipLimit=2)
            if len(np.shape(vis))==3:           
                current_gray=cv2.cvtColor(vis,cv2.COLOR_BGR2GRAY)
            else:
                current_gray=vis
            current_gray=current_gray[self.ROI[1]:(self.ROI[3]+self.ROI[1]),self.ROI[0]:(self.ROI[2]+self.ROI[0])] 

            p0 = cv2.goodFeaturesToTrack(current_gray, **self.feature_params)
            X=p0.reshape(-1, 2) 
            samples=[]
            for x in X:               
                featureExtraction=current_gray[int(x[0]-windowSize[0]/2):int(x[0]+windowSize[0]/2),int(x[1]-windowSize[1]/2):int(x[1]+windowSize[1]/2)]
                samples.append(featureExtraction)
                
            self.samplesMat.append(samples)  
            self.Xsamples.append(X)
            if display=='points':
                self.show(self,X,X,vis,display='points')

class videoAnalyser(Analyzer):
     def __init__(self,video_src,**args): 
        self.processingMode='video' 
        self.video_src=video_src                
        self.cap = cv2.VideoCapture(self.video_src)
        self.ret, self.frameInit = self.cap.read()   
        self.frameInit=Tools.convertToGrayScale(self.frameInit)
        self.number_of_frames=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        Analyzer.__init__(self,**args)



class frameSequenceAnalyzer(Analyzer):
    def __init__(self,**args): 
        self.processingMode='image sequence'        
        self.folder_src=args.get('folder_src','./')
        self.listD=np.sort(os.listdir(self.folder_src))     
        self.number_of_frames=len(self.listD)
        self.frameInit=cv2.imread(self.folder_src+'/'+self.listD[0])              
        self.frameInit=Tools.convertToGrayScale(self.frameInit)

        Analyzer.__init__(self,**args)

#    def writeGoodFeaturesPositionsAndExtraction(self,fileFormat='hdf5',imgFormat='png',outFolder='',filename=None,fileSequence=False): 
#        if filename is None:
#            filename='Goof_features_positions_from_frame'+ str(self.vec[0]) + 'to' + str(self.vec[-2]) + '_with_shift_'+str(self.paramVecTime['shift'])
#
#        for samples in self.samplesMat:
#           folderFrame=outFolder+'/'+filename+'/'+format(t,'04.0f')+'_to_'+format(t+self.paramVecTime['step']
#           Files.mkdir2()
#           cv2.imwrite()

           
    
                
                
        