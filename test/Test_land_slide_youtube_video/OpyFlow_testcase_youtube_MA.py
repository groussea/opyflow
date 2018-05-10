#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 08:24:12 2017

@author: Gauthier ROUSSEAU
"""


import sys, os
folder_main='.'
os.chdir(folder_main)
#Where is the opyf folder
sys.path.append('../')
import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt     
import vtk
import json

sys.path.append('/media/gauthier/Data_1/TAF/OPyF-Project/github/opyFlow')
import opyf
plt.rc('text', usetex=False)

#command lines To save the online youtube  video of the land slide (need the pytube package)
# pip install pytube on the cmd prompt

#from pytube import YouTube
#folder_main='/media/gauthier/Data_1/TAF/OPyF-Project/github/opyFlow/Test_land_slide_youtube_video'
#yt=YouTube("https://www.youtube.com/watch?time_continue=2&v=5-nyAz484WA")
#stre=yt.streams.first()
#stre.download(folder_main)

#%%

plt.close('all')


vidName='Val Parghera Zeitraffer Mai 2013 - Januar 2015.mp4'
#load video and deduce infos 

vidPath= folder_main+'/'+vidName
cap = cv2.VideoCapture(vidPath)
N=cap.get(cv2.CAP_PROP_FRAME_COUNT)
ret, frameini = cap.read()



#where are the images?


#Create a folder to save outputs (csv files, vtk files, images)
folder_outputs=folder_main+'/outputs'
opyf.mkdir2(folder_outputs)


#create an image output folder
folder_img_output=folder_outputs+'/images'
opyf.mkdir2(folder_img_output)


#Calculate the main proprieties of the frames

Hvis,Lvis,p=frameini.shape
ROI=[0,0,frameini.shape[1],frameini.shape[0]]
#It is possible to specify the ROI to accelerate the treatment
#ROI=[0,602,2799,1252]




#this fuctntion will produce 2 vectors
#select will select the index of the selected frame in listD
#prev is a boolean vector which indicate if there is an image to consider before or not


#Parameters for the Good Feature to Track algorithm (Shi-Tomasi Corner Detector) 
#the more we consider corners, the more we are able to reproduce the velocity
#be careful that whith a too low quality level for vectors the results are poor
#normal filters are needed to exclude bad vectors
#he nin distance is the minimum distance between vectors

feature_params = dict( maxCorners = 40000,
                       qualityLevel = 0.001,
                       minDistance = 5,
                       blockSize = 16)

#Parameters for the flow calculation using Lukas Kanade method
#WinSize caracterise the size of the window in which we search movement
#Warning : the algorithm is pyramidal. For the first step the 
lk_params = dict( winSize  = (16, 16),
                  maxLevel = 4,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.06))




# =============================================================================
# Parameters to filter outliers :
# 'vmin' and 'vmax' determine during the lk_algortihm which vector are directly 
# upressed from the calcultation of velocity.If it is None there is no effect
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


mask=cv2.imread(folder_main+'/mask.png')  

mask=mask[:,:,0]/255


filters_params = dict(vmin=None, 
                      vmax=None,
                      RadiusF=20.,
                      minNperRadius=4.,
                      maxDevinRadius=2.5,
                      DGF=5.)

#To interpolate datas we use vtk librarie which provide in its last versions efficient 
#Interpolation calculation
#
interp_params= dict (Radius=15.,
                     Sharpness=2.,
                     kernel='Gaussian',
                     scaleInterp=2)


#Many rendering are possible here are 3 important parameters.
# the lgortihm can plot 'horizontal', 'vertical' or norme field values
#
#for the vectos 
render_params= dict(Ptype='norme',
            vlim=[0,80],
            scale=1000)


#We define here the resolution for the interpolation
#the scale is usally higher (a resolution a bit lower than the image) but its 
#also possible to increase this resolution for special needs

framecut=frameini[ROI[1]:(ROI[3]+ROI[1]),ROI[0]:(ROI[2]+ROI[0])]
scaleinterp=interp_params['scaleInterp']
[Ly,Cx,z]=framecut.shape
resX=Cx/scaleinterp
resY=Ly/scaleinterp

grid_y, grid_x = np.mgrid[0:resY, 0:resX]
grid_x=(grid_x+0.5)*np.float(ROI[2])/resX+ROI[0]
grid_y=(grid_y+0.5)*np.float(ROI[3])/resY+ROI[1]

gridVx,gridVy=np.zeros_like(grid_x),np.zeros_like(grid_x)


#Save the parameters into a json dictionnary file

fulldict={'lk_params':lk_params,'feature_params':feature_params,
          'filters_params':filters_params,
          'interp_params':interp_params,'render_params':render_params}

out_file_name=folder_img_output+'/parametres.json'

out_file = open(out_file_name,"w")
# (the 'indent=4' is optional, but makes it more readable)
json.dump(fulldict,out_file, indent=4)                                    
out_file.close()

#Initialisation de la matrice pour la moyenne glissante (au saute 2)
ini_frame=0
nframe=20
amount_of_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
frames=np.zeros((Hvis,Lvis,nframe),'uint8')
for i in range(0,2*nframe,2):
    print(i)
    cap.set(cv2.CAP_PROP_POS_FRAMES,ini_frame+i)
    ret, frame = cap.read()
    frames[:,:,i//2] = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
frames_ini=frames    
frameav_glissante_ini=np.uint8(np.mean(frames,axis=2))

frameav_glissante=frameav_glissante_ini
frames=frames_ini

#fig1=plt.figure('track',figsize=(Lfig, Hfig))
#%%
prev_gray=None
tracks=[]
Xdata=np.empty([0,2])
Vdata=np.empty([0,2])
Xnul=[]
Vnul=[]
cap = cv2.VideoCapture(vidPath)

prev_gray=None
incr=0        
for i in range(ini_frame,np.int(N),2):
    print(np.float(i)/N)
    cap.set(cv2.CAP_PROP_POS_FRAMES, i+nframe/2 )
    ret, frame = cap.read()    
    frameC=frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, i+nframe)
    ret, frame = cap.read()    
    
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#Substract the moving averaged frame on nframe
    k=incr%nframe
    incr+=1
    frames[:,:,k]=frame
    frameav_glissante=np.mean(frames,axis=2)
    frame=frames[:,:,(incr-nframe/2 )%nframe]
    framet = cv2.cvtColor(frameC,cv2.COLOR_BGR2GRAY)
    framet=framet-0.5*frameav_glissante
    framet[np.where(framet<0)]=0
    framet=np.uint8(framet)
#        plt.imshow(framet)
    frame=framet
#    frame=opyf.Render.CLAHEbrightness(frame,10,tileGridSize=(32,32),clipLimit=3) 
    #Good feature + flow calculatition    
    prev_gray,X,V=opyf.Track.opyfFlowGoodFlag(frame,prev_gray,feature_params,
                                              lk_params,ROI=ROI,vmax=filters_params['vmax'],vmin=filters_params['vmin'],
                                              csvTrack=folder_outputs+'/'+format(incr,'04.0f')+'.csv',
                                              mask=mask,DGF=filters_params['DGF'])

#filtres les vecteurs abérents, plus de uniformité, tu cregardes une zone autour d'un vecteur et tu considères l'écart type autour du vecteur 
#pour identifier les outliers (le bruit), ça peut etre néfaste car des fois il y a des zones ou t'as bcp de bruit mais que tu les filtres quand même


    if len(X)>0:
        #filters
        Dev,Npoints,stD=opyf.Filters.opyfFindPointsWithinRadiusandDeviation(X,(V[:,0]**2+V[:,1]**2),filters_params['RadiusF'])
        X=opyf.opyfDeletePointCriterion(X,Npoints,climmin=filters_params['minNperRadius']) 
        V=opyf.opyfDeletePointCriterion(V,Npoints,climmin=filters_params['minNperRadius'])
        Dev,Npoints,stD=opyf.Filters.opyfFindPointsWithinRadiusandDeviation(X,V[:,1],filters_params['RadiusF'])
        X=opyf.opyfDeletePointCriterion(X,Dev,climmax=filters_params['maxDevinRadius']) 
        V=opyf.opyfDeletePointCriterion(V,Dev,climmax=filters_params['maxDevinRadius'])
        Dev,Npoints,stD=opyf.Filters.opyfFindPointsWithinRadiusandDeviation(X,V[:,0],filters_params['RadiusF'])
        X=opyf.opyfDeletePointCriterion(X,Dev,climmax=filters_params['maxDevinRadius']) 
        V=opyf.opyfDeletePointCriterion(V,Dev,climmax=filters_params['maxDevinRadius'])

    
    Norme=(gridVx**2+gridVy**2)**(0.5)
    if len(X)>0:
        Xdata=np.append(Xdata,X,axis=0)
        Vdata=np.append(Vdata,V,axis=0)
        
        TargetPoints=opyf.npGrid2TargetPoint2D(grid_x,grid_y)
    
        VT=opyf.npInterpolateVTK2D(X,V,TargetPoints,ParametreInterpolatorVTK=interp_params)
        gridVx=opyf.npTargetPoints2Grid2D([VT[:,0]],resX,resY)
        gridVy=opyf.npTargetPoints2Grid2D([VT[:,1]],resX,resY)

#set Plot is dictionnary that determine which type of plot we want
#it is possible to superimpose informations
        setPlot={'DisplayVis':True,
             'DisplayField':False,
         'QuiverOnFieldColored':False,
         'QuiverOnField':False,
         'DisplayPointsColored':True,
         'DisplayPoints':False,
         'QuiverOnPointsColored':False,
         'QuiverOnPoints':True,
         'DisplayContour':False,
         'ScaleVectors':None,
         'Dim':[Hvis,Lvis],
         'unit':'px'}      

#        setPlot={'DisplayVis':False,
#             'DisplayField':False,
#         'QuiverOnFieldColored':False,
#         'QuiverOnField':False,
#         'DisplayPointsColored':False,
#         'DisplayPoints':False,
#         'QuiverOnPointsColored':False,
#         'QuiverOnPoints':False,
#         'Dim':[Hvis,Lvis],
#         'unit':'px'}     
        render_params= dict(Ptype='norme',
            vlim=[0,80],
            scale=500)
        
        fig,ax=opyf.Render.opyfPlot(grid_x,grid_y,gridVx,gridVy,X,V,setPlot,vis=frame,**render_params) 
        fig.set_size_inches(12*1.5, 7*1.5)
        plt.savefig(folder_img_output+'/image_'+format(i,'04.0f')+'.png',format='png',dpi=200) 
#        figp,ax,qv,sm=opyf.Render.opyfQuiverPointCloudColored(Xdata,Vdata,ax=ax,** infoPlotQuiverPointCloud)
        plt.pause(0.1)

#%% Perform filters on final vectors

opyf.Render.opyfPlot(grid_x,grid_y,gridVx,gridVy,Xdata,Vdata,setPlot,vis=frame,namefig='Total',**render_params)         
Dev,Npoints,stD=opyf.Filters.opyfFindPointsWithinRadiusandDeviation(Xdata,(Vdata[:,0]**2+Vdata[:,1]**2),20.)
Xdata1=opyf.Filters.opyfDeletePointCriterion(Xdata,Dev,climmax=2.) 
Vdata1=opyf.Filters.opyfDeletePointCriterion(Vdata,Dev,climmax=2.)
opyf.Render.opyfPlot(grid_x,grid_y,gridVx,gridVy,Xdata1,Vdata1,setPlot,vis=frame,namefig='Total1',**render_params)         
Dev,Npoints,stD=opyf.Filters.opyfFindPointsWithinRadiusandDeviation(Xdata1,Vdata1[:,0],20.)
Xdata2=opyf.Filters.opyfDeletePointCriterion(Xdata1,Dev,climmax=2.) 
Vdata2=opyf.Filters.opyfDeletePointCriterion(Vdata1,Dev,climmax=2.)

#setPlot={'DisplayVis':True,
#                 'DisplayField':False,
#         'QuiverOnFieldColored':False,
#         'QuiverOnField':False,
#         'DisplayPointsColored':False,
#         'DisplayPoints':False,
#         'QuiverOnPointsColored':True,
#         'QuiverOnPoints':False,
#         'DisplayContour':False,
#         'ScaleVectors':None,
#         'Dim':[Hvis,Lvis],
#         'ROI':ROI,
#         'unit':'px'}        

setPlot={'DisplayVis':False,
             'DisplayField':False,
         'QuiverOnFieldColored':False,
         'QuiverOnField':False,
         'DisplayPointsColored':False,
         'DisplayPoints':False,
         'QuiverOnPointsColored':False,
         'QuiverOnPoints':False,
         'DisplayContour':False,
         'ScaleVectors':None,
         'Dim':[Hvis,Lvis],
         'unit':'px'}           


opyf.Render.opyfPlot(grid_x,grid_y,gridVx,gridVy,Xdata2,Vdata2,setPlot,vis=frame,nvec=5000,namefig='Total2',**render_params)         

fig=plt.gcf()
fig.set_size_inches([6,3.5])
ax=fig.axes[0]
ax.set_position([0.15,0.15,0.8,0.9])
ax.set_ylim([2000,500])
ax=fig.axes[1]
ax.set_position([0.15,0.12,0.8,0.05])
plt.savefig(folder_img_output+'/meanflow.png',format='png',dpi=200)              

#%% Perform Interpolation
if len(Xdata)>0 and l==listD[select[-1]]:

    
    TargetPoints=opyf.npGrid2TargetPoint2D(grid_x,grid_y)
    
    VT=opyf.npInterpolateVTK2D(Xdata2,Vdata2,TargetPoints,ParametreInterpolatorVTK=interp_params)
    gridVx=opyf.npTargetPoints2Grid2D([VT[:,0]],resX,resY)
    gridVy=opyf.npTargetPoints2Grid2D([VT[:,1]],resX,resY)
    
    Norme=(gridVx**2+gridVy**2)**(0.5)
    
#%%
    #Rendering stuff
    
setPlot={'DisplayVis':True,
                 'DisplayField':True,
         'QuiverOnFieldColored':False,
         'QuiverOnField':True,
         'DisplayPointsColored':False,
         'DisplayPoints':False,
         'QuiverOnPointsColored':False,
         'QuiverOnPoints':False,
         'DisplayContour':False,
         'ScaleVectors':None,
         'Text':True,
         'Dim':[Hvis,Lvis],
         'ROI':ROI,
         'unit':'px'} 

#setPlot={'DisplayVis':False,
#             'DisplayField':False,
#         'QuiverOnFieldColored':False,
#         'QuiverOnField':False,
#         'DisplayPointsColored':False,
#         'DisplayPoints':False,
#         'QuiverOnPointsColored':False,
#         'QuiverOnPoints':False,
#         'DisplayContour':False,
#         'ScaleVectors':None,
#         'Dim':[Hvis,Lvis],
#         'unit':'px'}   
cmap=opyf.setcmap('norme',alpha=0.5)
fig,ax=opyf.Render.opyfPlot(grid_x,grid_y,gridVx,gridVy,Xdata2,Vdata2,setPlot,vis=frame,namefig='Total2',vlim=[0,50],scale=1000,cmap=cmap)         

opyf.Render.opyfQuiverField(grid_x,grid_y,gridVx,gridVy,ax=ax)

opyf.Render.opyfText(fulldict,ax=ax,pos=(250,250))

plt.savefig(folder_img_output+'/field.png',format='png',dpi=100)              

        


     
    
    
