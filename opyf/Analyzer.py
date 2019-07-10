#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 08:59:41 2019

@author: gauthier
"""

import os
import numpy as np
import cv2
from opyf import MeshesAndTime
from opyf import Track
from opyf import Render
from opyf import Files
import matplotlib.pyplot as plt
class frameSequenceAnalyzer():
    def __init__(self,**args): 
        self.folder_src=args.get('folder_src','./')
        self.listD=np.sort(os.listdir(self.folder_src))     
        self.number_of_frames=len(self.listD)
        frameInit=cv2.imread(self.folder_src+'/'+self.listD[0],cv2.IMREAD_ANYDEPTH)
        self.Hvis,self.Lvis=frameInit.shape
        self.ROI=[0,0,frameInit.shape[1],frameInit.shape[0]]
        
        
        self.feature_params = dict( maxCorners = 40000,
                           qualityLevel = 0.0008,
                           minDistance = 3,
                           blockSize = 10)
    
    
        self.lk_params = dict( winSize  = (15, 15),
                          maxLevel = 2,
                          criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.03))
        
        self.filters_params = dict(vmin=None, 
                              vmax=None,
                              RadiusF=20.,
                              minNperRadius=2.,
                              maxDevinRadius='nean',
                              DGF=10)
        
        self.mask=np.ones(frameInit.shape)
        self.Xdata=[]
        self.Vdata=[]
        self.samplesMat=[]
        self.Xsamples=[]
        self.paramPlot={'ScaleVectors':0.1,
              'extentFrame':[0,self.Lvis,self.Hvis,0],
              'unit':['px','deltaT'],
              'vlim':[0,20],
              'Hfig':8}  

        
        
        self.setVecTime(**args)
        

    def setVecTime(self,**args):
        self.paramVecTime={'framedeb':args.get('framedeb',0),
                            'step':args.get('step',1),
                            'shift':args.get('shift',1),
                            'Ntot':args.get('Ntot',2)} 
         

        self.vec,self.prev=MeshesAndTime.setVecTime(framedeb=self.paramVecTime['framedeb'],
                                                     step=self.paramVecTime['step'],
                                                     shift=self.paramVecTime['shift'],
                                                     Ntot=self.paramVecTime['Ntot'])
        print('[The following image processing plan has been set]')
        for pr,i in zip(self.prev,self.vec):
            if pr==False:
                print('--> detect Good Features to Track on image [' +self.listD[i]+']')
                file_prev=self.listD[i]
            else:
                print('--> measure diplacements between image [' +file_prev +'] and [' +self.listD[i] +']')
        self.Time=self.vec[0:-1:2]        
    
    
    def extractGoodFeaturesPositionsAndDisplacements(self,display=False):
        
        for pr,i in zip(self.prev,self.vec):
            if pr==False:
                prev_gray=None
            #    frame=frame[:,:,1]

            l=self.listD[i]

            vis=cv2.imread(self.folder_src+'/'+l) #special for tiff images

   
            vis=Render.CLAHEbrightness(vis,0,tileGridSize=(20,20),clipLimit=2)
            if len(np.shape(vis))==3:           
                current_gray=cv2.cvtColor(vis,cv2.COLOR_BGR2GRAY)
            else:
                current_gray=vis
            current_gray=current_gray[self.ROI[1]:(self.ROI[3]+self.ROI[1]),self.ROI[0]:(self.ROI[2]+self.ROI[0])] 

                      
            prev_gray,X,V=Track.opyfFlowGoodFlag(current_gray,
                                                 prev_gray,
                                                 self.feature_params,
                                                 self.lk_params,
                                                 ROI=self.ROI,
                                                 vmax=self.filters_params['vmax'],
                                                 vmin=self.filters_params['vmin'],
                                                 mask=self.mask,
                                                 DGF=self.filters_params['DGF'])    
            
            if len(X)>0:
                print('From frame ['+ self.listD[i-self.paramVecTime['step']] +'] to frame [' + self.listD[i] +']')
                print('Number of Good Feature To Track = '+str(len(X)))
                print('Velocity max = '+str(np.max(V)))
                self.Xdata.append(X)
                self.Vdata.append(V)
                self.show(X,V,vis,display=display)
                


    def writeGoodFeaturesPositionsAndDisplacements(self,fileFormat='hdf5',outFolder='',filename=None,fileSequence=False):
        if filename is None:
            filename='Goof_features_positions_and_displacements_from_frame'+ str(self.vec[0]) + 'to' + str(self.vec[-1]) + '_with_step_' +str(self.paramVecTime['step']) + '_and_shift_'+str(self.paramVecTime['shift'])
        if fileFormat=='hdf5':
            Files.hdf5_WriteUnstructured2DTimeserie(outFolder+'/'+filename+'.'+fileFormat,self.Time,self.Xdata,self.Vdata)
        elif fileFormat=='csv':
            if fileSequence==False:
                Files.csv_WriteUnstructured2DTimeserie(outFolder+'/'+filename+'.'+fileFormat,self.Time,self.Xdata,self.Vdata)
            elif fileSequence==True:    
                Files.mkdir2(outFolder+'/'+filename)
                for x,v,t in zip(self.Xdata,self.Vdata,self.Time):
                    Files.write_csvTrack2D(outFolder+'/'+filename+'/'+format(t,'04.0f')+'_to_'+format(t+self.paramVecTime['step'],'04.0f')+'.'+fileFormat,x,v)
                
#            for x,v in zip(Xdata,Vdata):
    def extractGoodFeaturesPositionsAndExtraction(self,display=False,windowSize=(16,16)): 

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


#    def writeGoodFeaturesPositionsAndExtraction(self,fileFormat='hdf5',imgFormat='png',outFolder='',filename=None,fileSequence=False): 
#        if filename is None:
#            filename='Goof_features_positions_from_frame'+ str(self.vec[0]) + 'to' + str(self.vec[-2]) + '_with_shift_'+str(self.paramVecTime['shift'])
#
#        for samples in self.samplesMat:
#           folderFrame=outFolder+'/'+filename+'/'+format(t,'04.0f')+'_to_'+format(t+self.paramVecTime['step']
#           Files.mkdir2()
#           cv2.imwrite()

           
    def show(self,X,V,vis,display='quiver'):
        if display=='quiver':
        
          
            self.opyfDisp=Render.opyfDisplayer(**self.paramPlot)

            self.opyfDisp.plotQuiverUnstructured(X,V,vis=vis,width=0.005,normalize=True,alpha=1,c='cyan',nvec=7000)
            plt.pause(0.1)
        if display=='points':
        
          
            self.opyfDisp=Render.opyfDisplayer(**self.paramPlot)

            self.opyfDisp.plotPointsUnstructured(Xdata=X,Vdata=V,vis=vis,width=0.01,c='cyan',alpha=1,nvec=7000)
            plt.pause(0.1)
                
                
        