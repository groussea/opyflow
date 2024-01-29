
#%%
import sys
# sys.path.append('/folder/toward/opyf') (eventually add directly the opyf package folder if not installed)
import opyf
import matplotlib.pyplot as plt
# plt.ion()
import os
os.chdir("./")
import matplotlib
matplotlib.use('TkAgg')
plt.ion()
plt.close('all')
#Path toward the video file
filePath = './2018.07.04_Station_fixe_30m_sample.mp4'
#set the object information
video = opyf.videoAnalyzer(filePath)
#%%

video.set_trackingFeatures(
    Ntot=10, step=1, starting_frame=1, track_length=5, detection_interval=10)
'''
# the method {extractTracks} is available to extract tracks on images. The principle is inspired by the openCv sample lktrack.py (https://github.com/opencv/opencv/blob/master/samples/python/lk_track.py)
#The main difference rely on the possibility to store the tracks 
#Note that, if the object is build from a video with videoAnalyzer, a lag is expected since each required images a loaded in the memory for efficiency reasons
It might be a technic to better filter relevant velocities since it is possible to follow these patterns for multiple frames 
'''
opyf.mkdir2('./export_Tracks/')
video.set_filtersParams(wayBackGoodFlag=1, CLAHE=False)
video.extractTracks(display='quiver', displayColor=True,
                    saveImgPath='./export_Tracks/', numberingOutput=True)
# tracks can be saved in csv file
video.writeTracks(outFolder='./export_Tracks')
#when wrtiting it is possible to specify the out folder

# %%
