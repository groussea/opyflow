#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 21:25:08 2019

@author: gauthier Rousseau
"""
import numpy as np

def setVecTime(framedeb=0,step=1,shift=1,Ntot=2):
    deltaN=Ntot*shift    
    vec1=np.arange(int(framedeb),int(framedeb+deltaN),shift)
    vec2=np.arange(int(framedeb),int(framedeb+deltaN),shift)+step
    vec=np.array([vec1,vec2])
    l,c=vec.shape
    vec=np.reshape(np.array([vec1,vec2]),(1,l*c),order=1)
    vec=vec[0]
    prev=np.arange(0,len(vec))%2==1
   
    return vec,prev
