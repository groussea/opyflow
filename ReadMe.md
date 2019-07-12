
## About

OpyFlow : Python package for Optical FLow measurements.

It is based on openCV and VTK libraries to detect good features to track, calculate the optical flow by Lukas Kanade method and interpolate them on a mesh (see explanation below). The package contains also some rendering tools built with matplotlib tecplot, vtk, hdf5 readers and writers.
For flow calculations, it is mainly inspired on the openCV python sample lktrack.py.

Author: Gauthier Rousseau

This package has been developed in the course of my PhD at EPFL to study [Turbulent flows over rough permeable beds](https://infoscience.epfl.ch/record/264790/files/EPFL_TH9327.pdf). Outputs are visible in the manuscript and also on this [Video](https://www.youtube.com/watch?v=JmwE-kL0kTk) where videos and paraview animations have been rendered thanks to opyf outputs.

corresponding e-mail : gauthier.rousseau@gmail.com


"""

## Quick start

Assuming that you already have an environment with python installed (<=3.7), run the following command from the package repository:

''python setup.py install''

This should install opyf library and the main dependencies (vtk and opencv) automatically.

After you may test:

python /test/Test_case_PIV_Challenge_2014/CommandLines-Opyf-PIV-Challenge2014-Test_Simple.py

For any problem, contact me please.


## Contents

This archive is organized as follow:

The setup file:

--setup.py

The package Folder opyf:
--opyf
    --Track.py
    --Interpolate.py
    --Files.py
    --Filters.py
    --Render.py
    --custom_cmap.py
    --__init__.py


   
The test Folder:
--test
--Test_case_PIV_Challenge_2014
   --CommandLines-Opyf-PIV-Challenge2014-Test.py
   --CommandLines-Opyf-PIV-Challenge2014-Test_Simple.py
   --mask.tiff
   --images (sample of 2 source images)
--A_00001_a.tif
--A_00001_b.tif
--ReadMe_Download_Images.txt (instruction to download the other images of the test)
   --meanFlow.png (Results for the CommandLines)
   --rms.png
--Test_land_slide_youtube_video
   --OpyFlow_testcase_youtube_MA.py
   --OpyFlow_testcase_youtube_simple.py
   --mask.png
   --The video must be downloaded from youtube with the package pytube
   --ReadMe_download_a_youtube_video.txt (instruction to download the video)


Basic command lines:
--CommandLines-Opyf-Default.py



One test file performed on the PIV challenge 2014 caseA (images onhttp://www.pivchallenge.org/pivchallenge4.html#case_a):
The results are compared to the main findings of the challenge:
``-Kähler CJ, Astarita T, Vlachos PP, Sakakibara J, Hain R, Discetti S, Foy RL, Cierpka C, 2016, Main results of the 4th International PIV Challenge, Experiments in Fluids, 57: 97.''

A test on synthetic images is still required.

# Usage

## Installation

The package requires python and basic python package: csv, numpy, matplotlib, tqdm

The main dependencies are :

OpenCV
VTK

The code use last versions of VTK and openCV.
However, pip (Python Package Index) doesn't have the last vtk yet.
It is also a bit tricky to compile the source directly from the vtk website.
However, the simplest way to install it is to use miniconda or anaconda and the last updated sources from conda-forge.

When miniconda/anaconda is installed type in the command prompt:

conda create -n opyfenv vtk opencv matplotlib scipy tqdm (spyder)
source activate opyfenv

These command lines will install the an environnement with python 3.6.
The soft has been developed on python 2.7 and modifications are still required.
And then run spyder or directly ipython in the command prompt

it is also possible to use anaconda software to use a GUI.


## Applications

It is possible to test the algorithm on test case A of the python challenge.

Tested on:
Python version: 2.7 and 3.6
VTK : 7.0.1 and +
opencv : 3.2 and +
numpy: 1.17
matplotlib : 2.0.0

## Citation

@PhdThesis{rousseau2019turbulent,
  title={Turbulent flows over rough permeable beds in mountain rivers: Experimental insights and modeling},
  author={Rousseau, Gauthier},
  year={2019},
  institution={EPFL}
}






