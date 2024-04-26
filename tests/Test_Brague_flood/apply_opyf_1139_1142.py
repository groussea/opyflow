#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on April 26 2020

@author: Gauthier Rousseau
"""


#%%
# %matplotlib qt5
import sys, os
os.chdir("./") # if run in the folder of the video
# os.chdir(os.path.dirname(os.path.abspath(__file__))) # set path regarding the path of the python file, only works on jupyter
import opyf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
plt.close('all')

#%%

os.chdir("./1139")
video=opyf.videoAnalyzer('IMG_1139.MOV')

#extract a a picture to build a mask
# cv2.imwrite('for_mask_1142.png',video.vis)


video.set_vecTime(Ntot=10,starting_frame=200)

video.set_interpolationParams(Sharpness=2)
video.set_goodFeaturesToTrackParams(qualityLevel=0.01)
# video.set_filtersParams(CLAHE=True, maxDevInRadius=2, minNperRadius=5, wayBackGoodFlag=1,RadiusF=15)

mask=cv2.imread('mask_1139.png')
A=mask>100
video.set_stabilization(mask=A[:,:,0],mute=False)
image_points = np.array([
                    (355,429), #left 
                    (1338,350), # right
                    (99, 562),     # left barrier
                    (1673, 364),     # right barrier
                ], dtype="double")
 
# tarrain points of particular points at the surface of the water.
model_points = np.array([
                    (30.13,-8.28,0) ,#left 
                    (32.88,-28.08,0),# right bank
                    (20.46, -4.47, 0.4),    # left front bank
                    (21.32, -27.14, 0.4),], dtype="double")  #
#put the origin at the left cormner of the bridge
abs_or=model_points[0]
model_points = model_points -model_points[0]

video.set_birdEyeViewProcessing(image_points,model_points, [-12, 4, -32.],rotation=np.array([[1., 0, 0],[0,-1,0],[0,0,-1.]]),scale=True,framesPerSecond=30)

video.set_vlim([0, 10])

video.extractGoodFeaturesDisplacementsAccumulateAndInterpolate(display1='quiver',display2='field',displayColor=True)
video.set_filtersParams(maxDevInRadius=1.5, RadiusF=0.15,range_Vx=[0.01,10])

video.filterAndInterpolate()

#%% save accumulated points and velocities

X1139=np.copy(video.Xaccu+abs_or[0:2])
V1139=np.copy(video.Vaccu)
norm1139=(V1139[:, 0]** 2 + V1139[:, 1]** 2)** 0.5

#%% perfomed measurments on 1142
os.chdir("./../1142")
video=opyf.videoAnalyzer('IMG_1142.MOV')

video.set_vecTime(Ntot=10,starting_frame=0)
video.set_goodFeaturesToTrackParams(qualityLevel=0.001)

video.set_interpolationParams(Sharpness=2)
video.set_goodFeaturesToTrackParams(qualityLevel=0.001)
# video.set_filtersParams(CLAHE=True, maxDevInRadius=2, minNperRadius=5, wayBackGoodFlag=1)

mask=cv2.imread('mask_1142.png')
A=mask>100
video.set_stabilization(mask=A[:,:,0],mute=False)
image_points = np.array([
                    (830,564), #left arch
                    (1480, 594),     # right barrier
                    (1750,800),  # right front bush
                    (0,616), #left front bank
                    (369,565),  #left bank/ bridge 
                ], dtype="double")
 
# tarrain points of particular points at the surface of the water.
model_points = np.array([
                    (-2,-10.5,0) ,#left arch
                    (0, 0., 0),    # right barrier
                    (21.2, -5, 0), # right front bush
                    (7.4,-21.7,0),  #left front bank
                    (-2.6,-19.9,0), #left bank/ bridg                 
])   
video.set_birdEyeViewProcessing(image_points,
model_points, [12, 7, -30.],rotation=np.array([[1., 0, 0],[0,-1,0],[0,0,-1.]]),
scale=True,framesPerSecond=30)
video.set_vlim([0,10])
video.extractGoodFeaturesDisplacementsAccumulateAndInterpolate(display1='quiver', display2='field', displayColor=True)
video.set_filtersParams(maxDevInRadius=1.5, RadiusF=0.15,range_Vx=[0.01,10])


video.filterAndInterpolate()

X1142=np.copy(video.Xaccu)
V1142=np.copy(video.Vaccu)
norm1142=(V1142[:,0]**2+V1142[:,1]**2)**0.5
#%% scatter
plt.close('all')
fig, ax =plt.subplots(1,1)
ax.scatter(X1142[:,0],X1142[:,1],c=norm1142,alpha=0.3)
ax.scatter(X1139[:, 0], X1139[:, 1], c=norm1139, alpha=0.3)

ax.axis('equal')

plt.show()
#%% join data 
Xtot=np.append(X1142,X1139,axis=0)
Vtot=np.append(V1142, V1139,axis=0)
Ntot=np.append(norm1142, norm1139,axis=0)


#%% save data
export_H5='export_1142_1139.h5'
opyf.hdf5_WriteUnstructured2DTimeserie(export_H5,[0], [Xtot],[Vtot] )

#%% read data

[t], [Xtot],[Vtot]=opyf.hdf5_ReadUnstructured2DTimeserie(export_H5)
Ntot = (Vtot[:,0]**2+Vtot[:,1]**2)**0.5


#%% load MNT (can be downloaded here https://drive.switch.ch/index.php/s/cWnga5fAlucoZug)
import csv
f = open('MNT.xyz', 'r')
reader = csv.reader(f, delimiter='\t')
#    reader  = csv.reader(f,  dialect='excel')
index = 0
header = []
datas = []
nextIt = int(np.random.uniform() * 20)
for row in reader:
    if index==nextIt :
        temp = [float(item) for number, item in enumerate(row)]
        datas.append(temp)
        nextIt +=1+ int(np.random.uniform() * 99)
    index+=1

datas=np.array(datas)   

x0=1030760.6875
y0 = 6289057

terrain=np.copy(datas)
terrain[:, 0] = terrain[:, 0] - x0
terrain[:, 1] = terrain[:, 1] - y0
terrD = np.array([terrain[:, 2], terrain[:, 2]]).T



#%% construct bathy from the points

X1 = np.array([16., -2])
X2 = np.array([11., -23.])    

from numpy import linalg as LA
Npoints=1000
dr=LA.norm(X2 - X1)/Npoints
dvecr = (X2 - X1) / LA.norm(X1 - X2) * dr
ir = np.arange(0, Npoints)
vecX=np.array([X1 +dvecr*ir for ir in range(Npoints)])
normVecX=np.arange(Npoints)*dr
parm_interp = {'kernel': 'Gaussian',
                                    'Radius': 5*dr,
                                    'Sharpness': 2.}
V1 = opyf.Interpolate.npInterpolateVTK2D(terrain[:, 0:2], terrain[:, 1:3], vecX, ParametreInterpolatorVTK=parm_interp)
#%
plt.close('all')
plt.plot(normVecX,V1[:, 1])


#%%
bathy=np.zeros((len(V1),3))
bathy[:, 0:2] = vecX[:,0:2]
bathy[:, 2:3] = V1[:, 1:2]
# set the first point at the origin

plt.plot(bathy[:,1],bathy[:,2],'+')




OR_L=np.array([bathy[0,0],bathy[0,1]])
OR_R=np.array([bathy[-1,0],bathy[-1,1]])



#%%
plt.figure()
from scipy.interpolate import interp1d
fz = interp1d(bathy[:,1],bathy[:,2],fill_value="extrapolate")
# fy = interp1d(bathy[:,1],bathy[:,1])
XY_new= np.array([X1 +dvecr*(ir-100) for ir in range(Npoints+200)])
Y_new= XY_new[:,1]
bathy_new=np.zeros((len(Y_new),3))
bathy_new[:,2]=fz(Y_new)
bathy_new[:,1]=Y_new
bathy_new[:,0]=XY_new[:,0]
plt.plot(Y_new, bathy_new[:, 2], '+')
zwater=14.4 # m 14.2+- 0.2 m
bathy_new[:,2]=bathy_new[:,2]-zwater


#%% draw picture in the background


import rasterio as rio

filePath='Ortho.tif'
with rio.open(filePath) as img :
    imagea= img.read()
    imgmeta=img.meta

(cha,la, ca) = imagea.shape
image = np.zeros((la, ca, cha),dtype=np.uint8)

for cim in range(cha):
    image[:,:,cim]=imagea[cim,:,:]

del(imagea)
(la, ca,cha) = image.shape
scale=4
img_red = cv2.resize(image, (int(ca / scale), int(la / scale)), interpolation=cv2.INTER_AREA)
img_scale =0.00245*scale
(l_img, c_img, ch_img) = img_red.shape
shiftX, shiftY = -14.9397, 14.2917
y1=np.flipud((np.arange(l_img)-l_img/2)*img_scale) - shiftY
x1 = (np.arange(c_img) - c_img / 2) * img_scale- shiftX
vis=opyf.CLAHEbrightness(np.flipud(img_red[:,:,0:3]),10)

Xtotcopy=np.copy(Xtot)
bathy_newcopy=np.copy(bathy_new)
#%%
Xtot=np.copy(Xtotcopy)
bathy_new=np.copy(bathy_newcopy)
#%%
Xtot=np.copy(Xtotcopy)
bathy_new=np.copy(bathy_newcopy)
bathy_copy=np.copy(bathy)
OR_L=np.array([bathy[0,0],bathy[0,1]])
Xor=[0,OR_L[1]]
Xtot[:, 1]=-Xtot[:, 1]+Xor[1]
bathy_new[:,1]=-bathy_new[:,1]+Xor[1]
bathy_copy[:,1]=-bathy_copy[:,1]+Xor[1]
OR_L=np.array([bathy_copy[0,0],bathy_copy[0,1]])
OR_R=np.array([bathy_copy[-1,0],bathy_copy[-1,1]])
y1=np.flipud((np.arange(l_img)-l_img/2)*img_scale) - shiftY
x1 = (np.arange(c_img) - c_img / 2) * img_scale- shiftX
y1=-y1+Xor[1]
#%% Calcule du débit
# %matplotlib qt5

plt.close('all')

fig, ax = plt.subplots(1, 1, figsize=(7, 4))
cmapm=opyf.make_cmap_customized()

imsc=ax.scatter(Xtot[:, 0], Xtot[:, 1], c=Ntot, alpha=0.6,s=1,cmap=cmapm)

ax.axis('equal')

ax.plot([OR_R[0],OR_L[0]],[OR_R[1],OR_L[1]],'-x',ms=20,color=[.1,1,0.1,0.5],linewidth=8,markeredgewidth=4)
ax.plot([OR_R[0],OR_L[0]],[OR_R[1],OR_L[1]],'--x',ms=15,color=[0.,0.,0.,1],linewidth=2,markeredgewidth=1.5)
ax.text(OR_L[0]-4,OR_L[1]+0.1,'L',color='w',fontsize=18,bbox=dict(facecolor='purple',zorder=3, alpha=0.6))
ax.text(OR_R[0]-4,OR_R[1]-0.3,'R',color='w',fontsize=18,bbox=dict(facecolor='purple', zorder=3, alpha=0.6))

plt.rcParams['text.usetex'] = True
ParametreInterpolatorVTK = {'kernel': 'Gaussian',
                                    'Radius': 2,
                                    'Sharpness': 2.}

V=opyf.Interpolate.npInterpolateVTK2D(Xtot,Vtot,bathy_new[:,0:2],ParametreInterpolatorVTK)
norm = (V[:, 1]** 2 + V[:, 0]** 2)** 0.5
axR = fig.add_axes([0.3, 0.3, 0.63, 0.45],alpha=0.7)
bbox = dict(boxstyle="round", ec="w", fc="w", alpha=0.7)
plt.setp(axR.get_xticklabels(), bbox=bbox)
plt.setp(axR.get_yticklabels(), bbox=bbox)
bathy_z=np.array([float(b[2]) for b in bathy_new])

ind_im=np.where(bathy_z<=0.)

axR.plot(bathy_new[ind_im[0],1],V[ind_im[0],0],'--',linewidth=1,label='Norm Velocity ( in m/s )' )

axR.plot(bathy_new[ind_im[0], 1],bathy_z[ind_im[0]],'-.',linewidth=1.5,label='Bathymetry ( in m )' )
[x,y,X,Y]=axR.get_position().bounds
axR.xaxis.set_label_coords(0.4, 0.5)
axR.xaxis.label.set_backgroundcolor((1, 1, 1, 0.5))
axR.spines['bottom'].set_position(('data', 0))
axR.spines['left'].set_position(('data', 10))
axR.spines['right'].set_color('none')
axR.spines['top'].set_color('none')
axR.set_yticks([ -3, -2, -1, 1,2, 3,4  ])
axR.set_xticks([  0, 5, 15, 20   ])
axR.set_ylim([-3.8, 4.2])
plt.rcParams['xtick.major.size'] = 4
plt.rcParams['xtick.minor.size'] = 2
plt.rcParams['ytick.major.size'] = 4
plt.rcParams['ytick.minor.size'] = 2

axR.grid()
axR.minorticks_on()
axR.set_xlabel('Y [ m ]')
axR.patch.set_alpha(1)
ysup=bathy_new[ind_im[0][0], 1]
axR.xaxis.set_label_coords(0.67,0.38)

yinf=bathy_new[ind_im[0][-1], 1]
xim1=bathy_new[ind_im[0][0],0:2]
xim2=bathy_new[ind_im[0][-1],0:2]
lengthTrans=LA.norm(xim1-xim2)
Q=-np.mean(1*(V[ind_im,0]**2+V[ind_im,1]**2)**0.5*(bathy_z[ind_im]))*lengthTrans
axR.legend(fontsize=9,loc=6,bbox_to_anchor=(0.1, -0.25, 0.4, 0.2))
axR.text(13, 3, 'Q = ' + format(Q, '.1f') + ' m$^3$ s$^{-1}$', fontsize=10)
axR.text(0, -3,'L',color='w',fontsize=18,bbox=dict(facecolor='purple',zorder=3, alpha=0.6))
axR.text(22, -3,'R',color='w',fontsize=18,bbox=dict(facecolor='purple', zorder=3, alpha=0.6))
axR.set_xlim([0, 22])

axR.set_position([0.64, 0.22, 0.29, 0.7])
ax.set_position([0.1, 0.17, 0.4, 0.75])
axc=fig.add_axes([.53, .15, 0.01, 0.75])
ax.set_xlim([5, 25])
ax.set_ylim([24, -2])
ax.set_xlabel('X [ m ]')
ax.set_ylabel('Y [ m ]')
ax.imshow(vis,extent=[x1[0],x1[-1], y1[0], y1[-1]],alpha=0.9)

colc = fig.colorbar(imsc, cax=axc, label=r'$ \| \vec{U} \| $ [ m/s ]')
fig.show()
fig.set_size_inches((6, 3))
fig.savefig('Velocity_Bathymetry_Flow_1139_1142.png',dpi=200)


