#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 15:46:23 2018

@author: Gauthier Rousseau
"""


from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "ReadMe.md"),"r") as f:
    long_description = f.read()

setup(
    name="opyf",

    version="1.2",
    author="Gauthier Rousseau",
    author_email="gauthier.rousseau@gmail.com",

    description="OpyFlow - Python package for Optical Flow measurements.",
    
    long_description=long_description,
    long_description_content_type="text/markdown",
    # The project"s main homepage.
    url="https://github.com/groussea/opyflow",


    # Choose your license
    license="GPL-3.0",

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",

        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3"
    ],

    # What does your project relate to?
    keywords="optical flow",

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=["opyf"],

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed.
    install_requires=["ipython", "vtk", "opencv-python",
                      "tqdm", "h5py", "matplotlib", "scipy", "pytube"]
)
