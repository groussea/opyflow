#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 21:25:08 2019

@author: Gauthier Rousseau
"""
import numpy as np


def set_vecTime(starting_frame=0, step=1, shift=1, Ntot=2):
    vec=np.array([starting_frame+shift*int(i/2)+step*(i%2) for i in range(Ntot*2)])
    prev = np.arange(0, len(vec)) % 2 == 1
    return vec, prev


def set_vecTimeTracks(starting_frame=0, step=1,  Ntot=2):
    vecTracks=np.array([starting_frame+(step)*i for i in range(Ntot)])
    prevTracks=np.array([False]+[True for i in range(Ntot-1)])
    
    return vecTracks, prevTracks




def set_gridToInterpolateOn(pixLeft, pixRight, stepHor, pixUp, pixDown, stepVert):
    grid_y, grid_x = np.mgrid[pixUp:pixDown:stepVert, pixLeft:pixRight:stepHor]
    grid_y=grid_y+stepVert//2
    grid_x=grid_x+stepHor//2
    grid_Vx, grid_Vy = np.zeros_like(grid_y), np.zeros_like(grid_x)
    Hgrid, Lgrid = grid_y.shape
    return grid_y, grid_x, grid_Vy, grid_Vx, Hgrid, Lgrid
