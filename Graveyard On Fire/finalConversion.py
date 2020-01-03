#!/usr/local/bin/python3

import cv2
import numpy as np

npzFile = np.load('cam01distorted.npz')
dist = npzFile['dist']
mtx = npzFile['mtx']
newcameramtx = npzFile['newcameramtx']
roi = npzFile['roi']
rotM = npzFile['rotM']
tvec = npzFile['tvec']
cameraPosition = npzFile['cameraPosition']
pxlp = npzFile['pxlp']
realp = npzFile['realp']


z      = 0.0
camMat = np.asarray(mtx)
iRot   = np.linalg.inv(rotM)
iCam   = np.linalg.inv(camMat)

uvPoint = np.ones((3, 1))

# Image point
#uvPoint[0, 0] = extBot[0]
#uvPoint[1, 0] = extBot[1]

uvPoint[0, 0] = 237
uvPoint[1, 0] = 307

tempMat = np.matmul(np.matmul(iRot, iCam), uvPoint)
tempMat2 = np.matmul(iRot, tvec)

s = (z + tempMat2[2, 0]) / tempMat[2, 0]
wcPoint = np.matmul(iRot, (np.matmul(s * iCam, uvPoint) - tvec))

# wcPoint[2] will not be exactly equal to z, but very close to it
assert int(abs(wcPoint[2] - z) * (10 ** 8)) == 0
wcPoint[2] = z

print('wcp', wcPoint.T[0])






