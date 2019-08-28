# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 10:09:55 2017

@author: Gauthier ROUSSEAU 
"""

import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt


def opyfTrack(tracks, frame, prev_gray, incr, feature_params, lk_params, **args):
    X = np.empty([0, 1, 2])
    V = np.empty([0, 1, 2])
    mask = args.get('mask', None)
    csvTrack = args.get('csvTrack', None)
    vmin = args.get('vmin', -np.inf)
    if vmin == None:
        vmin = -np.inf
    vmax = args.get('vmax', np.inf)
    if vmax == None:
        vmax = np.inf
    DGF = args.get('DGF', 1)
    ROI = args.get('ROI', None)
#    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(40,40))
    if len(np.shape(frame)) == 3:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif len(np.shape(frame)) == 2:
        frame_gray = frame
    vis = frame_gray.copy()

    if len(tracks) > 0:
        img0, img1 = prev_gray, frame_gray
        p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            img0, img1, p0, None, **lk_params)
        p0r, st, err = cv2.calcOpticalFlowPyrLK(
            img1, img0, p1, None, **lk_params)
        d = abs(p0-p0r).reshape(-1, 2).max(-1)
        delta = abs(p1-p0).reshape(-1, 2).max(-1)
        good = np.logical_and((d < DGF), (delta > vmin), (delta < vmax))
        new_tracks = []

        for tr, (x, y), good_flag in zip(tracks, p1.reshape(-1, 2), good):
            if not good_flag:
                continue

            if int(y) >= frame.shape[0] or int(x) >= frame.shape[1] or int(y) < 0 or int(x) < 0:
                continue
            if mask is not None:
                if mask[int(y), int(x)] == 0:
                    continue
            tr.append((x, y))
            if len(tr) > 300.:
                del tr[0]
            new_tracks.append(tr)
            cv2.circle(vis, (x, y), 2, (255, 255, 0), -1)
        tracks = new_tracks
        X = np.float32([tr[-1] for tr in tracks])
        V = np.float32([tr[-1] for tr in tracks]) - \
            np.float32([tr[-2] for tr in tracks])
        Npart = np.arange(len(X))

    if incr % 50. == 0:
        mask1 = np.zeros_like(frame_gray)
        mask1[:] = 255
        for x, y in [np.int32(tr[-1]) for tr in tracks]:
            cv2.circle(mask1, (x, y), 200, 0, -1)
        p = cv2.goodFeaturesToTrack(frame_gray, mask=mask1, **feature_params)
        if p is not None:
            for x, y in np.float32(p).reshape(-1, 2):
                tracks.append([(x, y)])

    return tracks, frame_gray, X, V


def opyfFlow(frame, prev_gray, feature_params, lk_params, **args):
    X = []
    V = []
    mask = args.get('mask', None)
    csvTrack = args.get('csvTrack', None)
    vmin = args.get('vmin', -np.inf)
    vmax = args.get('vmax', np.inf)
    if np.shape(frame)[2] == 3:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif np.shape(frame)[2] == 1:
        frame_gray = frame
    p = cv2.goodFeaturesToTrack(frame_gray, **feature_params)
    if prev_gray is not None:
        img0, img1 = prev_gray, frame_gray
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            img0, img1, p, None, **lk_params)
        Xtemp = p.reshape(-1, 2)
        Vtemp = p1.reshape(-1, 2)-p.reshape(-1, 2)
        # Add preliminar filters

        for [vx, vy], [x, y] in zip(Vtemp, Xtemp):
            norme = (vx**2+vy**2)**0.5
            if norme < vmin or norme > vmax:
                continue
            if mask is not None:
                if mask[int(y), int(x)] == 0:
                    continue
            X.append([x, y])
            V.append([vx, vy])
        Npart = np.arange(len(X))
#        csvTrack=folder_outputs+'/'+format(incr,'04.0f')+'.csv'
        if csvTrack is not None:
            f = open(csvTrack, 'w')
            writer = csv.DictWriter(f, fieldnames=['N', 'X', 'Y', 'Ux', 'Uy'])
            writer.writeheader()
            for Ni in Npart:
                writer.writerow(
                    {'N': Ni+1, 'X': X[Ni][0], 'Y': X[Ni][1], 'Ux': V[Ni][0], 'Uy': V[Ni][1]})
            f.close()
    X = np.array(X)
    V = np.array(V)

    return frame_gray, X, V


def opyfFlowGoodFlag(frame, prev_gray, feature_params, lk_params, **args):
    X = []
    V = []
    mask = args.get('mask', None)
    vmin = args.get('vmin', -np.inf)
    if vmin == None:
        vmin = -np.inf
    vmax = args.get('vmax', np.inf)
    if vmax == None:
        vmax = np.inf
    DGF = args.get('DGF', 1)
    ROI = args.get('ROI', None)
    if ROI is not None:
        frame = frame[ROI[1]:(ROI[3]+ROI[1]), ROI[0]:(ROI[2]+ROI[0])]
    if len(np.shape(frame)) == 3:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif len(np.shape(frame)) == 2:
        frame_gray = frame
    p0 = cv2.goodFeaturesToTrack(frame_gray, **feature_params)
    if prev_gray is not None:
        img0, img1 = prev_gray, frame_gray
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            img0, img1, p0, None, **lk_params)
        p0r, st, err = cv2.calcOpticalFlowPyrLK(
            img1, img0, p1, None, **lk_params)
        d = abs(p0-p0r).reshape(-1, 2).max(-1)
        good = (d < DGF)
        Xtemp = p0.reshape(-1, 2)
        Vtemp = p1.reshape(-1, 2)-p0.reshape(-1, 2)
        # Add preliminar filters

        for [vx, vy], [x, y], good_flag in zip(Vtemp, Xtemp, good):
            if not good_flag:
                continue
            norme = (vx**2+vy**2)**0.5
            if norme < vmin or norme > vmax:
                continue
            if mask is not None:
                if mask[int(y), int(x)] == 0:
                    continue
            X.append([x, y])
            V.append([vx, vy])

#        csvTrack=folder_outputs+'/'+format(incr,'04.0f')+'.csv'

    X = np.array(X)
    V = np.array(V)
    if ROI is not None and len(X) > 0:
        X[:, 0] = X[:, 0]+ROI[0]
        X[:, 1] = X[:, 1]+ROI[1]

    return frame_gray, X, V
