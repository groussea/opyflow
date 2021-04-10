
# About

OpyFlow : Python package for Optical Flow measurements

Opyflow is a basic image velocimetry tool to simplify your *video* or *frame sequences* processing.

It is based on `opencv` and `vtk` libraries to detect Good Features to Track (GFT), calculate their displacements by the Lukas Kanade method and interpolate them on a mesh. This method is sometimes called Feature Image Velocimetry or Feature Tracking. It is an alternative to the classical cross-correlation techniques employed in Particle Image Velocimetry (PIV). Compared to this technique, GFT+OpticalFlow may result in better performance when image qualities are poor for velocimetry, i.e. when velocity information on frames is non-uniform.

For flow calculations, the process is mainly inspired by the openCV python sample [lktrack.py](https://github.com/opencv/opencv/blob/master/samples/python/lk_track.py).

The package also contains some rendering tools built with matplotlib. Velocities can be exported (csv, tecplot, vtk, hdf5).

Author: Gauthier Rousseau

Corresponding e-mail : gauthier.rousseau@gmail.com

## Quick start

Assuming that you already have an environment with python installed (<=3.9), run the following command on your terminal:

```shell
pip install opyf
```

or from the *opyflow* repository

```shell
python setup.py install
```

This should install the opyf library and the main dependencies (vtk and opencv) automatically.

If you meet compatibility problems on your system, it is recommended to create an *environment* via *conda* installation (see bellow [installation with anaconda](#installation-with-anaconda)).

To analyze a frame sequence (*png*, *bmp*, *jpeg*, *tiff*) you may run the following script:

```python
import opyf
analyzer=opyf.frameSequenceAnalyzer("folder/toward/images")
```

For a video (*mp4*, *avi*, *mkv*, ... ):

```python
analyzer=opyf.videoAnalyzer("video/file/path")
```

To perform your first analyze run :

```python
analyzer.extractGoodFeaturesAndDisplacements()
```

opyf package contains two frames and one video for testing and practicing yourself:

- The two frames were extracted from the frame sequence of the Test case A of the *PIV Challenge 2014*

![PIV challenge](test/Test_case_PIV_Challenge_2014/gif/example_PIV_challenge.gif)

When applied to the entire dataset, It can produce the above result (see [Test PIV Challenge 2014 - Case A](test/Test_case_PIV_Challenge_2014/testPIVChallengeCaseA.md) for details on the procedure).

- The video is a bird eye view video of a stream river taken by a drone and from which surface velocities can be extracted ([see the following python file for the different possible procedures](test/Test_Navizence/test_opyf_Navizence.py) ).

![bird eye view Navizence](test/Test_Navizence/gif/example_Navizence_Drone.gif)

- A guide to obtain the free surface velocity in a river from two amateur video is also provided :  
[Stabilization, orthorectification (brid eye view transformation) and free surface velocity estimation on the Brague river with opyFlow](test/Test_Brague_flood/test_opyf_LSPIV_Brague.md)

[]()<img src="test/Test_Brague_flood/figure_Brague.png" width=700>

## Contents

This repository is organized as follows:

The setup file:

- setup.py

The package Folder opyf:

- opyf 
  - Analyzer.py
  - Track.py
  - Interpolate.py
  - Files.py
  - Filters.py
  - Render.py
  - Tools.py
  - custom_cmap.py (based on Chris Slocum file)

- test
  - Test_case_PIV_Challenge_2014
  - Test_land_slide_youtube_video
  - Test_Navizence
  - Test_Brague_flood


A test on synthetic images is still required.

## Installation with anaconda

The package requires python and basic python package: csv, numpy, matplotlib, tqdm, hdf5. Main dependencies are: OpenCV and VTK.

If the `pip install opyf` command above does not work for you, the simplest way to deal with incompatibilities is using miniconda or anaconda.

When miniconda/anaconda is installed you may create an environment (here called *opyfenv*). To create the environment type in the terminal:

```shell
conda create -n opyfenv python=3.6 vtk opencv matplotlib scipy tqdm (spyder)
```

This command line will install an environment with python 3.6 and the main dependencies.

You can access to your environment by typing:

```shell
source activate opyfenv
```

From there, you should be able to install opyf with `pip install opyf`.

For python 3.7 you should use the *conda-forge* channel:

```shell
conda create -n opyfenv python=3.7 vtk opencv matplotlib scipy tqdm (spyder) --channel conda-forge
```

Tested on:
Python version: 3.6 and +
VTK : 7.0.1 and +
opencv : 3.2 and +
numpy: 1.17 and +
matplotlib : 2.0.0 and +

## Citation

This package has been developed in the course of my PhD at EPFL to study [Turbulent flows over rough permeable beds](https://infoscience.epfl.ch/record/264790/files/EPFL_TH9327.pdf). Outputs are visible in the manuscript as well as in this [Video](https://www.youtube.com/watch?v=JmwE-kL0kTk) where paraview animations have been rendered thanks to opyf outputs.

@PhdThesis{rousseau2019turbulent,
  title={Turbulent flows over rough permeable beds in mountain rivers: Experimental insights and modeling},
  author={Rousseau, Gauthier},
  year={2019},
  institution={EPFL}
}

An article published in *Experiments in Fluids* is available in Open Access : [Scanning PIV of turbulent flows over and through rough porous beds using refractive index matching](https://link.springer.com/article/10.1007/s00348-020-02990-y)

Follow this [link](https://link.springer.com/article/10.1007/s00348-020-02990-y#appendices) to directly access to the annex of the article presenting the opyf algorithms and tests.

@article{rousseau2020scanning,
  title={Scanning PIV of turbulent flows over and through rough porous beds using refractive index matching},
  author={Rousseau, Gauthier and Ancey, Christophe},
  journal={Experiments in Fluids},
  volume={61},
  number={8},
  pages={1--24},
  year={2020},
  publisher={Springer}
}

Contributors : Hugo Rousseau, Mohamed Nadeem, LHE team and others

Credits for UAV video : Bob de Graffenried

Credits for 2019 flood video on Brague River (Biot, French Riviera) : Pierre Brigode
