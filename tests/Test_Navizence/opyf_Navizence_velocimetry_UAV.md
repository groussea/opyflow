# Feature Image Velocimetry using *opyf* on a UAV/Drone Video of a Mountain Stream

## Preamble

This tutorial explains the procedure for applying opyflow on a UAV/Drone video of the Navizence mountain river. The procedure corresponds to the example presented in Annex F of [Rousseau (2019)](https://infoscience.epfl.ch/record/264790). After following the steps, you will be able to generate an accumulation of feature velocities on a sequence and perform interpolation on a background grid.

![Bird's Eye View of Navizence](gif/example_Navizence_Drone.gif)

## Initialization

Follow this [link](https://github.com/groussea/opyflow) to install the opyflow python package.

### Work Tree and Analyzer Object

First, let's set up the work tree and create an analyzer object to perform the analysis. Define the output folder and the location where your process will be performed.

```python
import sys
# sys.path.append('/folder/toward/opyf') (eventually add directly the opyf package folder if not installed using pip)
import opyf
import matplotlib.pyplot as plt
plt.ion()
import os
os.chdir("./")

sys.path.append('../../')

plt.close('all')
```

Specify the path to the video file.

```python
filePath = './2018.07.04_Station_fixe_30m_sample.mp4'
```

Set the videoAnalyzer object.

```python
video = opyf.videoAnalyzer(filePath)
```

This creates an object `video` that contains information derived from the video file. If you are working with a frame sequence, use `opyf.frameSequenceAnalyzer(path)` and provide the path where the images are stored.

### Time Vector Setting

Next, let's set the time vector for processing.

```python
video.set_vecTime(Ntot=10, shift=1, step=2, starting_frame=20)
print(video.vec, '\n', video.prev)
```

Use the `.set_vecTime` method to define the processing plan. This method sets `video.vec` and `video.prev`, two vectors required for image processing. By default, it plans a processing of the first two images of the video or frame sequence and extracts velocities between these two images. The values in `video.vec` represent the frame indices, and `video.prev` indicates if a previous image has been processed.  `video.prev=False` means that no previous image is processed, and  `True` means that the flow will be measured using the pair `{False+True}` from the good features detected in the first image.

The parameters `[Ntot]`, `[shift]`, `[starting_frame]`, and `[step]` control the processing plan. `[Ntot]` specifies the total number of image pairs, `[shift]` specifies the shift between two pairs, `[starting_frame]` specifies the first image, and `[step]` specifies the number of images between two images of each pair. Note that increasing the `[step]` value will result in larger displacements. Also, if the object is built from a video using `videoAnalyzer`, there might be a slight lag due to loading the required images into memory for efficiency reasons.

This function also defines `video.Time`, which is the time vector at which velocity measurements are performed.

## First Run with No Optimization

Now, let's run the analysis without any optimization.

```python
video.extractGoodFeaturesAndDisplacements(
    display='quiver', displayColor=True, width=0.002)
```

The method `.extractGoodFeaturesAndDisplacements` applied to the `video` object will detect the "good features to track" and calculate the optical flow according to the processing plan defined by `set_vecTime`. The option `display='quiver'` displays the velocity vectors corresponding to the features, while `display='points'` shows only the positions. The `displayColor` option introduces

 a colormap corresponding to the velocity magnitude. You can use the usual arguments for plotting with `plt.quiver` for the quiver plot or `plt.scatter` for the points. We can observe that the processing plan is really not optimal.

## Norm Velocity Limits

For improving results, we can provide the expected velocity magnitudes and set the velocity limits.

```python
video.set_vlim([0, 30])
```

The `set_vlim` method defines `video.vlim` and indicates the range of displacement norms expected with the processing plan. This is closely linked to the `[step]` parameter in `set_vecTime` since step controls displacements magnitude. You can run the processing again and observe the difference.

```python
video.extractGoodFeaturesAndDisplacements(
    display='quiver', displayColor=True, width=0.002)
```

## Filters

Now, let's apply some filters to eliminate outliers.

```python
video.set_filtersParams(wayBackGoodFlag=4, RadiusF=20,
                        maxDevInRadius=1, CLAHE=True)
```

The `set_filtersParams` method allows you to set various filters:

- `wayBackGoodFlag`: Specifies the distance accepted between the initial feature position and the position calculated by using the displacement from A to B and then from B to A (where A and B are the pair on which optical flow is calculated).
- `CLAHE`: Enables Contrast Limited Adaptive Histogram Equalization, which enhances local contrasts. Set it to `True` to apply the filter.
- `RadiusF` and `maxDevInRadius`: Specify the maximum deviation expected within `RadiusF`. Deviation is calculated as `(x - mean(S)) / std(S)`, where `S` is the velocity values within `RadiusF`, and `x` is the value being tested. If `Dev > maxDev`, the value is deleted.

## Good Features to Track

You can specify the parameters for "[Good Features to Track](https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga1d6bb77486c8f92d79c8793ad995d541)" which is a function provided by the `opencv` package (more information available in [the `opencv` tutorial for good features to track algorithm Shi, Tomasi (1994)](https://docs.opencv.org/4.x/d4/d8c/tutorial_py_shi_tomasi.html)).

```python
video.set_goodFeaturesToTrackParams(maxCorners=50000, qualityLevel=0.001)
```

You can access these parameters using `video.feature_params`.


## Optical Flow

You may also specify the parameters for optical flow. The default parameters are set by `video.set_opticalFlowParams(winSize=(16, 16), maxLevel=3)`. For more information on optical flow, refer to the [lukas kanade optical flowOpenCV documentation](https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html).

```python
video.set_opticalFlowParams(maxLevel=3)
```

You can access the optical flow parameters using `video.lk_params`.

## Extract Feature Displacements and Interpolate

Extract the velocity field by interpolating the displacements of the "good features to track" on a field defined by the method `set_gridToInterpolateOn`. By default, this grid is set to `(pixLeft=0, pixRight=0, stepHor=2, pixUp=0, pixDown=0, stepVert=2)`.

Interpolated data with accumulation will be stored in `video.UxTot[k]` and `video.UyTot[k]` for velocities at `video.Time[k]`.

```python
video.extractGoodFeaturesPositionsDisplacementsAndInterpolate(display='field', displayColor=True, scale=80, width=0.005)
```

To improve the quality of the treatment and obtain a converged flow field, you can increase the number of frame pairs `Ntot`. Here, we set `Ntot` to 88 instead of 10 in the previous example. This allows for convergence of flow statistics over at least 3 seconds (see

 Annex F of Rousseau (2019) for more explanations).

You can also regulate the smoothness of the data by reducing the sharpness using the `interpolator` parameter.

```python
video.set_interpolationParams(Sharpness=10, Radius=40)
video.set_vecTime(starting_frame=20, step=2, shift=1, Ntot=10)
video.extractGoodFeaturesDisplacementsAccumulateAndInterpolate(display1='quiver', display2='field', displayColor=True, scale=200)
```

If you apply the method `extractGoodFeaturesDisplacementsAccumulateAndInterpolate`, only one field will be produced at the end of the processing. Setting `display1='quiver'` displays the "good features to track" at each time step, while `display2='field'` displays the final field interpolated on the grid from all the velocity vectors. This method is useful when the flow is expected to be permanent or quasi-static during an interval that contains several frames. It is also appropriate to obtain the average velocity field from a sequence.

Interpolated data with accumulation are stored in `video.UxTot[0]` and `video.UyTot[0]`.

## Save Data

To extract the field data in CSV format, use the following code:

```python
video.writeVelocityField(fileFormat='csv')
```

If `extractGoodFeaturesDisplacementsAccumulateAndInterpolate` is run, only one file will be generated for the field resulting from the accumulation of vectors. If `extractGoodFeaturesPositionsDisplacementsAndInterpolate` is run, a series of CSV files will be generated.

The format of the CSV file is: X, Y, Ux, Uy.

By default, the vertical axis Y is oriented downward for images.

To save data in smaller files, the most convenient format is HDF5.

```python
video.writeVelocityField(fileFormat='hdf5')
# Check if the file is readable
opyf.hdf5_Read(video.filename+'.hdf5')
```

## Scale Data

```python
video.scaleData(framesPerSecond=25, metersPerPx=0.02,
                unit=['m', 's'], origin=[0, video.Hvis])
```

The scaling function is used if you want to scale the data by providing the frames per second (`framesPerSecond`) and the length scale (`metersPerPx`). The method is irreversible, meaning there is no possibility to revert the scaling. However, you can continue to process the scaled data.

Here, the scale is set to 2 centimeters per pixel. The Y axis is now oriented upward. You can verify this by using the following code:

```python
video.showXV(video.X, video.V, display='points', displayColor=True)
```

or by visualizing the averaged velocity field:

```python
Field = opyf.Render.setField(video.UxTot[0], video.UyTot[0], 'norm')
video.opyfDisp.plotField(Field, vis=video.vis)
```

This will plot the resulting averaged field.

```python
video.set_vecTime(Ntot=10, shift=1, step=1, starting_frame=20)
video.extractGoodFeaturesDisplacementsAccumulateAndInterpolate(display2='field', displayColor=True, scale=200)
video.set_trackingFeatures(Ntot=10, step=1, starting_frame=1, track_length=5, detection_interval=10)
```


## Tracking

The `extractTracks` method is available for extracting tracks on images. The principle is inspired by the `lktrack.py` sample from OpenCV's repository ([lk_track.py](https://github.com/opencv/opencv/blob/master/samples/python/lk_track.py)). The main difference is the ability to store the tracks.

Note that if the object is built from a video using `videoAnalyzer`, there might be a lag since each required image is loaded into memory for efficiency reasons. It is a technique to better filter relevant velocities, as it is possible to track these patterns for multiple frames. Note also that if the vector field was accumulated over `Ntot` images, the track will only have `Ntot-1` positions.

```python
opyf.mkdir2('./export_Tracks/')
video.set_filtersParams(wayBackGoodFlag=1, CLAHE=False)
video.extractTracks(display='quiver', displayColor=True, saveImgPath='./export_Tracks/', numberingOutput=True)
```

Tracks can be saved in a CSV file:

```python
video.writeTracks(outFolder='./export_Tracks')
```

In the code above, the `mkdir2` function creates a directory called `export_Tracks` if it doesn't already exist. The `set_filtersParams` function is used to set filter parameters, such as the `wayBackGoodFlag` and `CLAHE` options. The `extractTracks` method extracts and displays the tracks, using the `display` parameter to visualize them as a quiver plot with color-coded velocities. The `saveImgPath` parameter specifies the path to save the exported images, and `numberingOutput` adds numbering to the output files. Finally, the `writeTracks` method saves the tracks in the specified output folder.
