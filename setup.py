#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 15:46:23 2018

@author: gauthier
"""

from distutils.core import setup

import numpy as np

def readme(fname):
    with open(fname, 'r') as f:
        return f.read()

setup(
    name = 'opyf',
    version = '0.1',
    description = 'Calculate optical flow on videos and image sequences',
    long_description = readme('ReadMe.txt'),
    author = 'Gauthier ROUSSEAU',
    author_email = 'gauthier.rousseau@gmail.com',
    url = 'https://github.com/groussea/opyflow',
    packages = ['opyf'],
)
