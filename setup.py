#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 15:46:23 2018

@author: gauthier
"""

#from distutils.core import setup

#import numpy as np

from setuptools import setup


#def readme(fname):
#    with open(fname, 'r') as f:
#        return f.read()
#
#setup(
#    name = 'opyf',
#    version = '0.1',
#    description = 'Calculate optical flow on videos and image sequences',
#    long_description = readme('ReadMe.txt'),
#    author = 'Gauthier ROUSSEAU',
#    author_email = 'gauthier.rousseau@gmail.com',
#    url = 'https://github.com/groussea/opyflow',
#    packages = ['opyf'],
#)

# Always prefer setuptools over distutils
from setuptools import setup, find_packages

# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'ReadMe.txt'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='opyf',

    # Versions should comply with PEP440.
    version='0.1',

    description='OpyFlow : Python package for Optical FLow measurements.',
    long_description=long_description,

    # The project's main homepage.
    url="https://github.com/groussea/opyflow",

    # Author details
    author='gauthier Rousseau',
    author_email='gauthier.rousseau@gmail.com',

    # Choose your license
    license='GPL-3.0',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries :: Python Modules',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],

    # What does your project relate to?
    keywords='development',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=['opyf'],

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed.
    install_requires=['ipython','vtk','opencv-python','tqdm','matplotlib','scipy','pytube']
)
