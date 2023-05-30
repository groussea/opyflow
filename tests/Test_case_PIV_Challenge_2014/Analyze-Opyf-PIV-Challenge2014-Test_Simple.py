#!/usr/bin/env python3
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
# Create a folder to save outputs (csv files, vtk files, hdf5 files, images)
folder_outputs = folder_main+'/outputs'
opyf.mkdir2(folder_outputs)

#For .tif files it is generally usefull to specify the range of Pixels (RangeOfPixels=[px_min,px_max]).
#FOr the PIV challenge case A, pixels range from 450 to 900 in the raw .tif frames.
#%
frames = opyf.frameSequenceAnalyzer(folder_src, imreadOption=2, rangeOfPixels=[450, 900])

frames.set_vlim([0,90])
# frames.extractGoodFeaturesDisplacementsAccumulateAndInterpolate()


frames.set_filtersParams(wayBackGoodFlag=1)
frames.extractGoodFeaturesPositionsDisplacementsAndInterpolate()
frames.opyfDisp.fig.savefig('/media/gauthier/Data-Gauthier/programs/gitHub/opyflow/test/Test_case_PIV_Challenge_2014/gif/example2.png', dpi=100)


frames.writeVelocityField(fileFormat='csv')
# frames.writeGoodFeaturesPositionsAndDisplacements(fileFormat='csv', outFolder=folder_outputs)



#%% A mask to discard some regions of the frames can b applied
# For the PIV CHallenge, an mask has been produced and is availaible in the PIV Challenge test folder. The mask can be set when defining the analyzer object or simply using the *set_mask* option.
frames = opyf.frameSequenceAnalyzer(folder_src, imreadOption=2, rangeOfPixels=[450, 900], vlim=[2, 60], mask=folder_main+'/mask.tiff')

# or if the frame ocject has already been produced: 

frames.set_mask(folder_main + '/mask.tiff')

frames.extractGoodFeaturesPositionsDisplacementsAndInterpolate()


#%% If you loaded the entire dataset, you may run the code bellow
#  (only the 2 first frames are in the images folder, it is required to download the entire dataset on the PIV Challenge website )

frames.set_vecTime(Ntot=600,shift=2)
frames.extractGoodFeaturesDisplacementsAccumulateAndInterpolate()
frames.set_filtersParams(wayBackGoodFlag=1.,RadiusF=5,maxDevInRadius=2.5)

# This will accumulate all the vectors in a unique list and the interpolate them on a background mesh 

#%% To generate the figure

frames.set_vecTime(Ntot=600,shift=2)
frames.extractGoodFeaturesDisplacementsAccumulateAndInterpolate()
frames.set_filtersParams(wayBackGoodFlag=1.,RadiusF=5,maxDevInRadius=2.5)
frames.Xaccu, frames.Vaccu=frames.applyFilters(frames.Xaccu, frames.Vaccu)
frames.interpolateOnGrid(frames.Xaccu, frames.Vaccu)
frames.Field[np.where(frames.Field==0)]=np.nan
frames.Field=frames.Field*frames.gridMask
frames.Field=opyf.Render.setField(frames.Ux, frames.Uy, 'norm')

 frames.opyfDisp.fig.savefig('/media/gauthier/Data-Gauthier/programs/gitHub/opyflow/test/Test_case_PIV_Challenge_2014/gif/test.png', dpi=142)
