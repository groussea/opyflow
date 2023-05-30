#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 15:48:39 2017

@author: Gauthier ROUSSEAU
"""

import vtk
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
# A function that allow to use interpolation algorithm of vtk
# Source Numpy array of points [X] and values [V]
# The Points are 2D numpy arrays and values need to be at same size
# TODO an home made interpolator from pointlocator vtk class  

def npInterpolateVTK2D(npPoints, npValues, npTargetPoints, ParametreInterpolatorVTK=None):
    # Set SParameters if not define
    
    if ParametreInterpolatorVTK == None:
        ParametreInterpolatorVTK = {'kernel': 'Gaussian',
                                    'Radius': 20.,
                                    'Sharpness': 2.}
    print('[' + ParametreInterpolatorVTK['kernel'] + ' Interpolation - Radius =' + str(ParametreInterpolatorVTK['Radius']
                                                                                       ) + ' - Sharpness =' + str(ParametreInterpolatorVTK['Sharpness']) + '] processing... ')
# Set Source Points
    UnGrid = vtk.vtkUnstructuredGrid()
    vtkP = vtk.vtkPoints()
    for [x, y] in npPoints:
        vtkP.InsertNextPoint(x, y, 0.)

    UnGrid.SetPoints(vtkP)
# Set source Point Values
    l, c = npValues.shape
    for i in range(0, c):
        vtkFA = vtk.vtkFloatArray()
        vtkFA.SetName('Values'+str(i))
        for v in npValues:
            vtkFA.InsertNextValue(v[i])
        UnGrid.GetPointData().AddArray(vtkFA)

# Set Target Points

    vtkTP = vtk.vtkPoints()
    for [x, y] in npTargetPoints:
        vtkTP.InsertNextPoint(x, y, 0.)

    vtkTargetPointsPolyData = vtk.vtkPolyData()
    vtkTargetPointsPolyData.SetPoints(vtkTP)

    if ParametreInterpolatorVTK['kernel'] == 'Gaussian':

        Kernel = vtk.vtkGaussianKernel()
        Kernel.SetSharpness(ParametreInterpolatorVTK['Sharpness'])
        Kernel.SetRadius(ParametreInterpolatorVTK['Radius'])

    if ParametreInterpolatorVTK['kernel'] == 'Voronoi':
        Kernel = vtk.vtkVoronoiKernel()

    if ParametreInterpolatorVTK['kernel'] == 'Shepard':
        Kernel = vtk.vtkShepardKernel()
        Kernel.SetRadius(ParametreInterpolatorVTK['Radius'])

    interp = vtk.vtkPointInterpolator2D()
    interp.SetInputData(vtkTargetPointsPolyData)
    interp.SetSourceData(UnGrid)
    interp.SetKernel(Kernel)
#        interp.GetLocator().SetNumberOfPointsPerBucket(1)
    interp.InterpolateZOff()
    interp.SetNullPointsStrategyToMaskPoints()

    interp.Update()

    outputInterp = interp.GetOutput()
    pointsArr = outputInterp.GetPoints().GetData()
    nppointsArr = vtk_to_numpy(pointsArr)
    pdata = outputInterp.GetPointData()

# Convert volocities into Numpy Array

    npOutputValues = np.zeros((len(npTargetPoints), c))

    for i in range(0, c):
        vtkOutputValues = pdata.GetArray('Values'+str(i))
        npOutputValues[:, i] = vtk_to_numpy(vtkOutputValues)

    return npOutputValues

# %%


def npInterpolateVTK3D(npPoints, npValues, npTargetPoints, ParametreInterpolatorVTK=None):

    if ParametreInterpolatorVTK == None:
        ParametreInterpolatorVTK = {'kernel': 'Gaussian',
                                    'Radius': 10.,
                                    'Sharpness': 2.}
    print('[' + ParametreInterpolatorVTK['kernel'] + ' Interpolation - Radius =' + str(ParametreInterpolatorVTK['Radius']
                                                                                       ) + ' - Sharpness =' + str(ParametreInterpolatorVTK['Sharpness']) + '] processing... ')
#
# Set Source Points
    UnGrid = vtk.vtkUnstructuredGrid()
    vtkP = vtk.vtkPoints()
    for [x, y, z] in npPoints:
        vtkP.InsertNextPoint(x, y, z)

    UnGrid.SetPoints(vtkP)
# Set source Point Values
    l, c = npValues.shape
    for i in range(0, c):
        vtkFA = vtk.vtkFloatArray()
        vtkFA.SetName('Values'+str(i))
        for v in npValues:
            vtkFA.InsertNextValue(v[i])
        UnGrid.GetPointData().AddArray(vtkFA)

# Set Target Points

    vtkTP = vtk.vtkPoints()
    for [x, y, z] in npTargetPoints:
        vtkTP.InsertNextPoint(x, y, z)

    vtkTargetPointsPolyData = vtk.vtkPolyData()
    vtkTargetPointsPolyData.SetPoints(vtkTP)

    if ParametreInterpolatorVTK['kernel'] == 'Gaussian':

        Kernel = vtk.vtkGaussianKernel()
        Kernel.SetSharpness(ParametreInterpolatorVTK['Sharpness'])
        Kernel.SetRadius(ParametreInterpolatorVTK['Radius'])

    if ParametreInterpolatorVTK['kernel'] == 'Voronoi':
        Kernel = vtk.vtkVoronoiKernel()

    if ParametreInterpolatorVTK['kernel'] == 'Shepard':
        Kernel = vtk.vtkShepardKernel()
        Kernel.SetRadius(ParametreInterpolatorVTK['Radius'])

   # Build locator
    locator = vtk.vtkStaticPointLocator()
    locator.SetDataSet(UnGrid)
    locator.BuildLocator()
    # build interpolator
    interp = vtk.vtkPointInterpolator()
    interp.SetInputData(vtkTargetPointsPolyData)
    interp.SetSourceData(UnGrid)
    interp.SetKernel(Kernel)
    interp.SetLocator(locator)
    interp.GetLocator().SetNumberOfPointsPerBucket(5)
    interp.SetNullPointsStrategyToMaskPoints()

    interp.Update()

    outputInterp = interp.GetOutput()
    pointsArr = outputInterp.GetPoints().GetData()
    nppointsArr = vtk_to_numpy(pointsArr)
    pdata = outputInterp.GetPointData()

# Convert volocities into Numpy Array

    npOutputValues = np.zeros((len(npTargetPoints), c))

    for i in range(0, c):
        vtkOutputValues = pdata.GetArray('Values'+str(i))
        npOutputValues[:, i] = vtk_to_numpy(vtkOutputValues)

    return npOutputValues

# %%


def npGrid2TargetPoint3D(npGridx, npGridy, npGridz):
    pts = np.empty(npGridz.shape + (3,), dtype=float)
    pts[..., 0] = npGridx
    pts[..., 1] = npGridy
    pts[..., 2] = npGridz
    pts = pts.transpose(2, 0, 1, 3).copy()
    pts.shape = pts.size // 3, 3
    return pts


def npGrid2TargetPoint2D(npGridx, npGridy):
    l, c = npGridx.shape
    TargetPoint = np.zeros((l*c, 2))
    TargetPoint[:, 0] = np.reshape(npGridx, (1, l*c))
    TargetPoint[:, 1] = np.reshape(npGridy, (1, l*c))
    return TargetPoint


def npGrid2TargetPoint2D1col(npGridx):
    l, c = npGridx.shape
    TargetPoint = np.zeros((l*c, 1))
    TargetPoint = np.reshape(npGridx, (1, l*c))
    return TargetPoint


def npTargetPoints2Grid2D(npTargetPoint, resX, resY):
    grid = np.reshape(npTargetPoint, (resY, resX))
    return grid


def npTargetPoints2Grid3D(npTargetPoint, resX, resY, resZ):
    grid = np.reshape(npTargetPoint, (resZ, resY, resX))
    grid = np.transpose(grid, (1, 2, 0))
    return grid


# Old Style interpolation


# def interpolate2D(points,Vel,grid_x,grid_y,paramInterp,mask=None):
#    if paramInterp['type']=='krigeage' or paramInterp['type']=='gaussian':
#        import libInterpFortran
#    else:
#        from scipy.interpolate import griddata
#
#    (Ny,Nx)=grid_x.shape
#    if mask is None:
#        mask=np.uint8(np.ones((Ny,Nx))*255)
#
#    grid_val=np.zeros((Ny,Nx))
#    xval=points[:,0]
#    yval=points[:,1]
#
#    if paramInterp['mod'] is 'norme':
#        values=np.array((Vel[:,0]**2+Vel[:,1]**2)**0.5)
#    else:
#        if paramInterp['mod']=='horizontal':
#            values=Vel[:,0]
#        elif paramInterp['mod']=='vertical':
#            values=Vel[:,1]
#        else:
#            print('error : not a good mod')
#
#    if len(xval)>10:
#        if paramInterp['type']=='krigeage':
#            Po=np.transpose(np.reshape([xval,yval,values],(3,-1)))
#            bw = paramInterp['bw']
#            Kmax = paramInterp['Kmax']
#            hval = np.arange(bw,Kmax,bw)
#            sv = libInterpFortran.empiricalsemivariogram(xval,yval,values,hval)
#            popt = libkrige.Kfit(libkrige.Kgauss,hval,sv)
#            plt.figure()
#            plt.plot(hval, sv, '.-' )
#            plt.plot(hval, libkrige.Kgauss(hval,popt[0],popt[1])) ;
#            plt.title('Exponential Model')
#            plt.ylabel('Semivariance')
#            plt.xlabel('Lag [px]')
#
#        if paramInterp['type']=='linear':
#            grid_val = griddata(points[:,0:2], values, (grid_x, grid_y), method='linear')
#        else:
#            for i in range(Nx):
#                print i,
#                for j in range(Ny):
#                    if mask[j,i]==0:
#                        grid_val[j,i]=float('nan')
#                    else:
#                        if paramInterp['type']=='gaussian':
#                            grid_val[j,i]=libInterpFortran.gaussianinterp1val(xval,yval,values,grid_x[0,i],grid_y[j,0],paramInterp['GaussR'],paramInterp['Gaussnpoints'])
#                        elif paramInterp['type']=='krigeage':
#                            grid_val[j,i]=libkrige.Krige(Po, libkrige.Kgauss,(np.float(i),np.float(j)),paramInterp['Knpoints'],popt)
#                        else:
#                            print('error : interpType = gaussian or krigeage or linear')
#                            break
#
#        grid_val[np.where(grid_val>10**3)]=float('nan')
#
#    return grid_val
