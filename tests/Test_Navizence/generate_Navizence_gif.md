# Generate gif or video from analysis

This page presents the basic command lines to obtain a gif or video using the [opyflow](/ReadMe.md) algortihm as below:

![bird eye view Navizence](/test/Test_Navizence/gif/example_Navizence_Drone.gif)


The code first generates the velocity vectors measured between several pairs of images, then interpolates the velocities on a grid to represent the average velocity magnitude. 


## Initialization

The code is expected to be run from the folder containing the video from which we want to create a gif (or a video) (here *2018.07.04_Station_fixe_30m_sample.mp4*)


```python
import sys, os
os.chdir("./")
# if opyf is not installed where is the opyf folder?
sys.path.append('../../') #here we add the opyf folder from the default *2018.07.04_Station_fixe_30m_sample.mp4* folder (opyf/test/Test_Navizence).
import opyf 
import matplotlib.pyplot as plt
import numpy as np
plt.close('all')
#Path toward the video file
filePath='./2018.07.04_Station_fixe_30m_sample.mp4'
#set the opyf videoAnalyzer object
video=opyf.videoAnalyzer(filePath)
```


## Generate images with vectors
```python
for s in range(20,50,5):
    video.set_vecTime(Ntot=2, step=2, shift=1, starting_frame=s)
    video.set_goodFeaturesToTrackParams(maxCorners=50000, qualityLevel=0.001)
    video.set_filtersParams(wayBackGoodFlag=4,RadiusF=2,maxDevInRadius=1,CLAHE=True)
    video.scaleData(framesPerSecond=25, metersPerPx=12.1/600, unit=['m', 's'], origin=[0,video.Hvis])
    video.set_vlim([0,5])
    video.extractGoodFeaturesDisplacementsAndAccumulate()
    video.opyfDisp.fig.set_size_inches(7, 4)
    # video.opyfDisp.plotField(video.Field,vlim=[0,60])
    # [optional] Modify the default position of the axe
    video.showXV(video.Xaccu, video.Vaccu, vis=video.vis, display='quiver',nvec=5000,vlim=[0,5])
    video.opyfDisp.ax.set_position([0.12232868757259015,
    0.24062988281250006,
    0.7953426248548199,
    0.6687402343749999])
    video.opyfDisp.ax.set_aspect('equal', adjustable='box')
    # set the dimension of the images
    video.opyfDisp.ax.set_xlabel('')
    # video.invertYaxis()
    video.opyfDisp.fig.set_size_inches(7, 4)
    plt.pause(0.5)
    video.opyfDisp.fig.savefig('frame'+str(s)+'.png', dpi=142)

```

```python
video.opyfDisp.ax.set_position([0.12232868757259015,
 0.24062988281250006,
 0.7953426248548199,
 0.6687402343749999])

video.opyfDisp.ax.set_aspect('equal', adjustable='box')
video.opyfDisp.fig.set_size_inches(7, 4)
video.opyfDisp.ax.set_xlabel('')
plt.show()
video.opyfDisp.fig.savefig('frame_final.png', dpi=142)
```


```python
video.Field = np.zeros_like(video.grid_x)
video.opyfDisp.ax.set_position([0.12232868757259015,
0.24062988281250006,
0.7953426248548199,
0.6687402343749999])
video.opyfDisp.ax.set_aspect('equal', adjustable='box')
video.opyfDisp.fig.set_size_inches(7, 4)
video.opyfDisp.ax.set_xlabel('')
video.Field[np.where(video.Field==0)]=np.nan
video.scaleData(framesPerSecond=25, metersPerPx=12.1/600, unit=['m', 's'], origin=[0,video.Hvis])
video.opyfDisp.plotField(video.Field, vis=video.vis, vlim=[0, 5])
video.opyfDisp.fig.savefig('/media/gauthier/Data-Gauthier/programs/gitHub/opyflow/test/Test_Navizence/gif/init.png', dpi=142)

```