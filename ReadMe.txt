# About

OpyFlow : Python package for Optical FLow measurements.
It is based on openCV and VTK libraries to detect good features, calculate the flow on pixels and interpolate them on mesh
(see explaination below). The package contain also some rendering tools built with matplotlib.


Authors: Gauthier Rousseau with the initial help of Hugo Rousseau and the recent help of Mohamed Nadeem

corresponding e-mail : gauthier.rousseau@gmail.com

## Contents

This archive is organized as follow:


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
--Test_case_PIV_Challenge_2014
    --CommandLines-Opyf-PIV-Challenge2014-Test.py
    --CommandLines-Opyf-PIV-Challenge2014-Test_Simple.py
    --mask.tiff
    --images (folder)
	--A_00001_a.tif
	--A_00001_b.tif	
	--ReadMe_Download_Images.txt (instruction to download the other images of the test)
    --meanFlow.png (Results for the CommandLines)
    --rms.png
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

The code use last versions of VTK and opencv.
However, pip (Python Package Index) doesn't have the last vtk yet.
It is also a bit tricky to compile the source directly from the vtk website. 
However, the simplest way to install it is to use miniconda oranaconda and the last updated sources from conda-forge.

When miniconda/anaconda is installed type in the commande prompt:

conda create -n opyfenv python=2.7 vtk opencv matplotlib scipy tqdm (spyder) 
source activate opyfenv

And then run spyder or directly ipython in the command prompt

it is also possible to use anaconda software to use a GUI.


## Applications

It is possible to test the algorithm on test case A of the python challenge.

tested on:
Python version: 2.7
VTK : 8.0.1
opencv : 3.2
numpy: 1.17
matplotlib : 2.0.0

