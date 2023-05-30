#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 14:58:58 2017

@author: Gauthier ROUSSEAU
"""
#%%

import vtk
import numpy as np
import time
from vtk.util.numpy_support import vtk_to_numpy
from tqdm import tqdm


def opyfBuildLocatorandStuff2D(X):
    # Create points array which are positions to probe data with
    # FindClosestPoint(), We also create an array to hold the results of this
    # probe operation and
    # This array is the same array since we want to find the closest neighbor
    # The strategy will be to find the 2 closest point since one of them will be the same point
    # And we consider the second one, the farthest
    npPoints = X
    points = vtk.vtkPoints()
    points.SetDataTypeToDouble()
    probePoints = vtk.vtkPoints()
    probePoints.SetDataTypeToDouble()

    for [x, y] in npPoints:
        points.InsertNextPoint(x, y, 0.)
        probePoints.InsertNextPoint(x, y, 0.)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    points.ComputeBounds()

    staticLocator = vtk.vtkStaticPointLocator()
    staticLocator.SetDataSet(polydata)
    staticLocator.SetNumberOfPointsPerBucket(5)
    staticLocator.AutomaticOn()

    staticLocator.BuildLocator()

    return staticLocator, points, probePoints


def opyfBuildLocatorandStuff3D(X):

    npPoints = X
    points = vtk.vtkPoints()
    points.SetDataTypeToDouble()
    probePoints = vtk.vtkPoints()
    probePoints.SetDataTypeToDouble()

    for [x, y, z] in npPoints:
        points.InsertNextPoint(x, y, z)
        probePoints.InsertNextPoint(x, y, z)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    points.ComputeBounds()

    staticLocator = vtk.vtkStaticPointLocator()
    staticLocator.SetDataSet(polydata)
    staticLocator.SetNumberOfPointsPerBucket(5)
    staticLocator.AutomaticOn()

    staticLocator.BuildLocator()

    return staticLocator, points, probePoints


def opyfFindClosestPointandDistance(X):

    staticLocator, points, probePoints = opyfBuildLocatorandStuff2D(X)

    staticClosestN = vtk.vtkIdList()
    ind = np.zeros(len(X), dtype=np.int)
    D = np.zeros(len(X))
    math = vtk.vtkMath()
    x = [0, 0, 0]
    p = [0, 0, 0]
    staticClosestN = vtk.vtkIdList()
    for i in range(len(X)):
        staticLocator.FindClosestNPoints(
            2, probePoints.GetPoint(i), staticClosestN)
        ind[i] = staticClosestN.GetId(1)  # we then select the furthest point
        points.GetPoint(ind[i], x)
        points.GetPoint(i, p)
        D[i] = math.Distance2BetweenPoints(x, p)**0.5

    return ind, D


def opyfFindClosestPointandDistance3D(X):

    staticLocator, points, probePoints = opyfBuildLocatorandStuff3D(X)

    staticClosestN = vtk.vtkIdList()
    ind = np.zeros(len(X), dtype=np.int)
    D = np.zeros(len(X))
    math = vtk.vtkMath()
    x = [0, 0, 0]
    p = [0, 0, 0]
    staticClosestN = vtk.vtkIdList()
    for i in range(len(X)):
        staticLocator.FindClosestNPoints(
            2, probePoints.GetPoint(i), staticClosestN)
        ind[i] = staticClosestN.GetId(1)  # we then select the furthest point
        points.GetPoint(ind[i], x)
        points.GetPoint(i, p)
        D[i] = math.Distance2BetweenPoints(x, p)**0.5

    return ind, D


# Exclude isolated points the strategy will be to exclude point at a distance bigger
#    than D=50

# Creation of a function which delete the vector lines if Crit>climmax or Crit<climmin.
# Crit is an array of criterion
def opyfDeletePointCriterion(Vect, Crit, climmin=-np.inf, climmax=np.inf):
    booln = (Crit < climmin) | (Crit > climmax)
    indDel = np.where(booln)

    newVect = np.delete(Vect, indDel, 0)

    return newVect

# Exclude outliers, exclude outliers that present statistically a big difference with neighbors


def opyfFindPointsWithinRadiusandDeviation(X, Value, R):

    staticLocator, points, probePoints = opyfBuildLocatorandStuff2D(X)

    Dev = np.zeros(len(X))
    Npoints = np.zeros(len(X), dtype=int)
    PointswhithinRadius = vtk.vtkIdList()
    stD = np.zeros(len(X))
    print('[I] Find Point within Radius processing may take a while if the number of points is too large)')
    print('[I] Find Points within Radius='+ format(R,'2.2f')  +'- Processing ---- ')
    for i in tqdm(range(len(X))):
        # if (i) % (np.round(len(X)/3.)) == 0:
            # print('[I]'+format((i)*100/len(X)+1,'2.0f')+'%')
        staticLocator.FindPointsWithinRadius(
            R, probePoints.GetPoint(i), PointswhithinRadius)
        tempind = []
        Npoints[i] = PointswhithinRadius.GetNumberOfIds()
        for j in range(PointswhithinRadius.GetNumberOfIds()):
            tempind.append(PointswhithinRadius.GetId(j))
        stD[i] = np.std(Value[tempind])
        if np.mean(Value[tempind]) == Value[i]:
            Dev[i] = 0
        else:
            Dev[i] = np.abs(Value[i]-np.mean(Value[tempind]))/stD[i]

    return Dev, Npoints, stD


def opyfFindPointsWithinRadiusandDeviation3D(X, Value, R):

    staticLocator, points, probePoints = opyfBuildLocatorandStuff3D(X)

    Dev = np.zeros(len(X))
    Npoints = np.zeros(len(X), dtype=int)
    PointswhithinRadius = vtk.vtkIdList()
    stD = np.zeros(len(X))
    print('[I] Find Point within Radius processing may take a while if the number of points is too large)')
    print('[I] Find Points within Radius='+ format(R,'2.2f')  +'- Processing ---- ')
    for i in tqdm(range(len(X))):
        # if (i) % (np.round(len(X)/3.)) == 0:
            # print('[I]'+format((i)*100/len(X)+1,'2.0f')+'%')
        staticLocator.FindPointsWithinRadius(R, probePoints.GetPoint(i), PointswhithinRadius)
        tempind = []
        Npoints[i] = PointswhithinRadius.GetNumberOfIds()
        for j in range(PointswhithinRadius.GetNumberOfIds()):
            tempind.append(PointswhithinRadius.GetId(j))
        stD[i] = np.std(Value[tempind])
        if np.mean(Value[tempind]) == Value[i]:
            Dev[i] = 0
        else:
            Dev[i] = np.abs(Value[i]-np.mean(Value[tempind]))/stD[i]

    return Dev, Npoints, stD


# %%
def opyfFindBlobs3D(scalars, th1, th2=None, R=1., minArea=None, CenterType='barycenter'):
    if th2 is None:
        th2 = np.max(scalars)

    [h, w, p] = scalars.shape
    scalars = scalars.T.copy()
    scalars = scalars.ravel()
#    (scalars>=th1)*(scalars<=th2)
#    indth=np.where((scalars>=th1)*(scalars<=th2))
    # Points XYZ
    x, y, z = np.mgrid[0:h, 0:w, 0:p]
    pts = np.empty(z.shape + (3,), dtype=float)
    pts[..., 0] = x
    pts[..., 1] = y
    pts[..., 2] = z
    pts = pts.transpose(2, 1, 0, 3).copy()
    pts.shape = pts.size / 3, 3

    # VTK
    points = vtk.vtkPoints()
    points.SetDataTypeToDouble()

    for [x, y, z] in pts:
        points.InsertNextPoint(x, y, z)

    Scalarsvtk = vtk.vtkFloatArray()

    for s in scalars:
        Scalarsvtk.InsertNextValue(s)
    Ids = vtk.vtkIdList()

    polydata = vtk.vtkPolyData()
#    polydata=vtk.vtkUnstructuredGrid()
    polydata.SetPoints(points)
    polydata.GetPointData().SetScalars(Scalarsvtk)

    ThresholdIn = vtk.vtkThresholdPoints()
    ThresholdIn.SetInputData(polydata)
    # ThresholdIn.ThresholdByLower(230.)
    ThresholdIn.ThresholdBetween(th1, th2)

    ThresholdIn.Update()

    PointsThresh = ThresholdIn.GetOutput().GetPoints()

    staticLocator = vtk.vtkStaticPointLocator()
    staticLocator.SetDataSet(ThresholdIn.GetOutput())
    staticLocator.SetNumberOfPointsPerBucket(5)
    staticLocator.AutomaticOn()

    staticLocator.BuildLocator()

    S = ThresholdIn.GetOutput().GetPointData().GetArray(0)
    nppointsArr = vtk_to_numpy(S)

    idspointsThresh = np.arange(0, PointsThresh.GetNumberOfPoints())
    entitiy = 0
    i = idspointsThresh[0]
    idsStored = np.array([])
    ids1 = [0]

    C = 1
    blob3D = []
    indstore = 0

    while len(idspointsThresh) > 0:
        C = 1
        while C == 1:
            ids2 = []
            l = len(idsStored)
            for i in ids1:
                PointswhithinRadius = vtk.vtkIdList()
                staticLocator.FindPointsWithinRadius(
                    R, PointsThresh.GetPoint(i), PointswhithinRadius)

                for j in range(PointswhithinRadius.GetNumberOfIds()):
                    tempid = PointswhithinRadius.GetId(j)
                    if len(np.where(idsStored == tempid)[0]) == 0:
                        idsStored = np.append(idsStored, np.int(tempid))
                        ids2.append(tempid)
                        indDel = np.where(idspointsThresh == tempid)
                        idspointsThresh = np.delete(idspointsThresh, indDel, 0)

            if len(idsStored) == l:
                C = 0
                blob3D.append(idsStored[indstore:].astype(int))
                indstore = len(idsStored)
                entitiy += 1
                if len(idspointsThresh) > 0:
                    ids1 = [idspointsThresh[0]]
            else:
                ids1 = ids2

    C = np.zeros((len(blob3D), 3))

    AreaBlob = []
    X = vtk_to_numpy(PointsThresh.GetData())
    blob3Dout = []
    for i in range(len(blob3D)):
        ind = blob3D[i]

        Xblob = X[ind]
        blob3Dout.append(Xblob)
        pxInts = nppointsArr[ind]
        if CenterType == 'barycenter':
            C[i, :] = np.sum(
                Xblob*np.array([pxInts, pxInts, pxInts]).T, axis=0)/(np.sum(pxInts))
        elif CenterType == 'geometric':
            C[i, :] = np.sum(Xblob, axis=0)/(np.float(len(pxInts)))
        AreaBlob.append(len(Xblob))
    AreaBlob = np.array(AreaBlob)
    blob3Dout = np.array(blob3Dout)
    if minArea is not None:
        C = C[np.where(AreaBlob > minArea)]
        blob3Dout = blob3Dout[np.where(AreaBlob > minArea)]
        AreaBlob = AreaBlob[np.where(AreaBlob > minArea)]

    return blob3Dout, C, AreaBlob


def opyfFindBlobs3D_structured(scalars, th1, th2=None, R=1., minArea=None, CenterType='barycenter'):

    # Usually slower than unstructured above with vtk tools
    if th2 is None:
        th2 = np.max(scalars)

    [h, w, p] = scalars.shape
#    (scalars>=th1)*(scalars<=th2)
#    indth=np.where((scalars>=th1)*(scalars<=th2))
    # Points XYZ
    x, y, z = np.mgrid[0:h, 0:w, 0:p]

    indth = np.array(np.where((scalars > th1)*(scalars < th2)))

    # VTK

    binarry = np.array(
        np.where((scalars > th1)*(scalars < th2), 1, 0), dtype='?')
    entitiy = 0
    i = indth[:, 0]
    idsStored = [tuple(i)]
    binarry[idsStored[0]] = False
    ids1 = [indth[:, 0]]
    C = 1
    blob3D = []
    indstore = 0
    kernel = [(-1, 0, 0), (1, 0, 0), (0, 1, 0),
              (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    incr = 0
    while len(indth) > 0:
        C = 1
        while C == 1:
            ids2 = []
            l = len(idsStored)
            for i in ids1:
                for ii in range(0, 6):
                    indtemp = tuple(i+kernel[ii])
                    if 0 <= indtemp[0] < h and 0 <= indtemp[1] < w and 0 <= indtemp[2] < p:
                        v = binarry[indtemp]
                        if v == True:
                            idsStored.append(tuple(i+kernel[ii]))
                            ids2.append(indtemp)
                            binarry[indtemp] = False
            indth = np.array(np.where(binarry == True))

            if len(idsStored) == l:
                C = 0
                blob3D.append(idsStored)
                entitiy += 1
                if len(indth[0]) > 0:
                    ids1 = [indth[:, 0]]
                    idsStored = [tuple(ids1[0])]
                    binarry[idsStored[0]] = False
            else:
                ids1 = np.array(ids2)
            incr += 1

    C = np.zeros((len(blob3D), 3))

    AreaBlob = []
    X = vtk_to_numpy(PointsThresh.GetData())
    blob3Dout = []
    for i in range(len(blob3D)):
        ind = blob3D[i]

        Xblob = X[ind]
        blob3Dout.append(Xblob)
        pxInts = nppointsArr[ind]
        if CenterType == 'barycenter':
            C[i, :] = np.sum(
                Xblob*np.array([pxInts, pxInts, pxInts]).T, axis=0)/(np.sum(pxInts))
        elif CenterType == 'geometric':
            C[i, :] = np.sum(Xblob, axis=0)/(np.float(len(pxInts)))
        AreaBlob.append(len(Xblob))
    AreaBlob = np.array(AreaBlob)
    blob3Dout = np.array(blob3Dout)
    if minArea is not None:
        C = C[np.where(AreaBlob > minArea)]
        blob3Dout = blob3Dout[np.where(AreaBlob > minArea)]
        AreaBlob = AreaBlob[np.where(AreaBlob > minArea)]

    return blob3Dout, C, AreaBlob


def findClosestPointsTwoLists(X_source,X_target):
    '''return the closest points indexes in X_source from the X_target points,
    indexes output has the same length than X_target'''

    listPoints = X_source
    points = vtk.vtkPoints()
    points.SetDataTypeToDouble()

    listProbePoints=X_target
    probePoints = vtk.vtkPoints()
    probePoints.SetDataTypeToDouble()



    for [x, y] in listPoints:
        points.InsertNextPoint(x, y, 0.)
    for [x, y] in listProbePoints:
        probePoints.InsertNextPoint(x, y, 0.)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    points.ComputeBounds()

    staticLocator = vtk.vtkStaticPointLocator()
    staticLocator.SetDataSet(polydata)
    staticLocator.SetNumberOfPointsPerBucket(5)
    staticLocator.AutomaticOn()

    staticLocator.BuildLocator()

    staticClosestN = vtk.vtkIdList()
    ind = np.zeros(len(X_target), dtype=np.int)
    D = np.zeros(len(X_target))
    math = vtk.vtkMath()
    x = [0, 0, 0]
    p = [0, 0, 0]
    staticClosestN = vtk.vtkIdList()
    for i in range(len(X_target)):
        staticLocator.FindClosestNPoints(
            1, probePoints.GetPoint(i), staticClosestN)
        ind[i] = staticClosestN.GetId(0)  # we then select the closest point
        points.GetPoint(ind[i], x)
        probePoints.GetPoint(i, p)
        D[i] = math.Distance2BetweenPoints(x, p)**0.5

    return ind, D