# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 10:09:55 2017

@author: Gauthier ROUSSEAU 
"""

import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt


def opyfTrack(tracks, vtracks, gray, prev_gray, incr, feature_params, lk_params, tracks_params, **args):
    print('Tracking processing...')
    X = np.empty([0, 1, 2])
    V = np.empty([0, 1, 2])
    maskFrame = args.get('mask', None)
    vmin = args.get('vmin', -np.inf)
    if vmin == None:
        vmin = -np.inf
    vmax = args.get('vmax', np.inf)
    if vmax == None:
        vmax = np.inf
    wayBackGoodFlag = args.get('wayBackGoodFlag', 1)
    ROI = args.get('ROI', None)

    if len(tracks) > 0 and prev_gray is not None:
        img0, img1 = prev_gray, gray
        p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            img0, img1, p0, None, **lk_params)
        p0r, st, err = cv2.calcOpticalFlowPyrLK(
            img1, img0, p1, None, **lk_params)
        d = abs(p0-p0r).reshape(-1, 2).max(-1)
        delta = abs(p1-p0).reshape(-1, 2).max(-1)
        good = np.logical_and((d < wayBackGoodFlag),
                              (delta > vmin), (delta < vmax))
        new_tracks = []
        new_vtracks = []
        Vtemp = p1.reshape(-1, 2)-p0.reshape(-1, 2)
        for tr, vtr, (x, y), (vx, vy), good_flag in zip(tracks, vtracks, p1.reshape(-1, 2), Vtemp, good):
            if not good_flag:
                continue
            if int(y) >= gray.shape[0] or int(x) >= gray.shape[1] or int(y) < 0 or int(x) < 0:
                continue
            norm = (vx**2+vy**2)**0.5
            if norm < vmin or norm > vmax:
                continue
            if maskFrame is not None:
                if maskFrame[int(y), int(x)] == 0:
                    continue
            tr.append((x, y))
            vtr.append((vx, vy))
            if len(tr) > tracks_params['track_len']:
                del tr[0]
                del vtr[0]
            new_tracks.append(tr)
            new_vtracks.append(vtr)
        tracks = new_tracks
        vtracks = new_vtracks
        print('---Number of tracks :'+str(len(tracks)))
        X = np.float32([tr[-1] for tr in tracks])
        V = np.float32([vtr[-1] for vtr in vtracks])

    if incr % tracks_params['detect_interval'] == 0:
        mask = np.zeros_like(gray)
        mask[:] = 255

        print('Good feature detection processing...')
        for x, y in [np.int32(tr[-1]) for tr in tracks]:
            cv2.circle(mask, (x, y), 5, 0, -1)
        p = cv2.goodFeaturesToTrack(gray, mask=mask, **feature_params)

        print('----- Number of new features detected :'+str(len(p)))
        if p is not None:
            for x, y in np.float32(p).reshape(-1, 2):
                tracks.append([(x, y)])
                vtracks.append([('nan', 'nan')])
    if len(X) == 0:
        X = np.empty((0, 2))
        V = np.empty((0, 2))
    X = np.array(X)
    V = np.array(V)
    if ROI is not None and len(X) > 0:
        X[:, 0] = X[:, 0]+ROI[0]
        X[:, 1] = X[:, 1]+ROI[1]
    return tracks, vtracks, gray, X, V


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
    # wayBackGoodFlag determine the acceptable error when flow is claculated from A to B
    wayBackGoodFlag = args.get('wayBackGoodFlag', 1)
    # and then from B to A.
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
        good = (d < wayBackGoodFlag)
        Xtemp = p0.reshape(-1, 2)
        Vtemp = p1.reshape(-1, 2)-p0.reshape(-1, 2)
        # Add preliminar filters

        for [vx, vy], [x, y], good_flag in zip(Vtemp, Xtemp, good):
            if not good_flag:
                continue
            norm = (vx**2+vy**2)**0.5
            if norm < vmin or norm > vmax:
                continue
            if mask is not None:
                if mask[int(y), int(x)] == 0:
                    continue
            X.append([x+vx/2, y+vy/2]) # position of the measured velocity at half of the displacement
            V.append([vx, vy])

#        csvTrack=folder_outputs+'/'+format(incr,'04.0f')+'.csv'
    if X == []:
        X = np.empty((0, 2))
        V = np.empty((0, 2))
    X = np.array(X)
    V = np.array(V)
    if ROI is not None and len(X) > 0:
        X[:, 0] = X[:, 0]+ROI[0]
        X[:, 1] = X[:, 1]+ROI[1]

    #return the intermediate position if velocities processed
    return frame_gray, X, V
