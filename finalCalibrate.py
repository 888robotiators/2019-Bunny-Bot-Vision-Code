#!/usr/local/bin/python3

#------------------------------------------------------------------------------
# FUNCTIONAL
#------------------------------------------------------------------------------
from collections import deque

import argparse
import cv2 as cv
import glob
import imutils
import math
import numpy as np


# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

#prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((9*7, 3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:7].T.reshape(-1,2)


#to inches
for (i, element) in enumerate(objp):
    objp[i] = element * 20.0 / 25.4
# Arrays to store object points and image points from all the images
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image space


images = glob.glob('*.jpg')

#go through all images
for fname in images:

    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (9,7), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)

        cv.drawChessboardCorners(img, (9,7), corners, ret)

        cv.imshow('img', img)

        cv.waitKey(100)

#gets the camera matrix (intrinsic parameters) and rotation and translation vectors (external parametres)
#mtx = 3x3, rvecs = 3x3, tvecs = 3x1
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
img = cv.imread('calibrateM01.jpg')
h, w = img.shape[:2]
newcameramtx, roi= cv.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

#calculate the error of the intrinsics
tot_error = 0
for i in xrange(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    tot_error += error

print("mean error: ", tot_error/len(objpoints))



#undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)

#crop the image
x,y,w,h = roi




#now find the extrinsic parameters of the camera (rvec and tvec)

#create numpy 4x3 array of zeroes in order to store 4 real world coordinates
realp = np.zeros((4, 3), np.float32)


# Arrays to store object points and image points from all the images
#top left, top right, bottom left, bottom right
realpoints = [] # 3d points of the corners in real world space
pixelpoints = [] # 2d points of the corners in pixel lotation


#dimensions of paper
height = 11.0
width  = 8.5

'''
#measured point in inches
realLeft = np.array([-9.5, 24.25, 0])
#testing different values to get best camera position
realLeft = np.array([-9.5 - 0.0625, 24.25 + 0.0832832529, 0])
realBottom = np.array([-5.91, 20.00 + 1.625, 0])

#in order to calculate other points...
angle = math.atan2(4.25, 9.625)
dx    = width * math.cos(angle)
dy    = width * math.sin(angle)

realLeft = np.array([realBottom[0] - dx, realBottom[1] + dy, 0])

angle = (math.pi / 2.0) - angle
dx    = height * math.cos(angle)
dy    = height * math.sin(angle)

#calculate other points
realTop   = np.array([realLeft[0]   + dx, realLeft[1]   + dy, 0])
realRight = np.array([realBottom[0] + dx, realBottom[1] + dy, 0])
'''

realLeft = [-20.25, 52, 0]
realTop = [-12, 86, 0]
realRight = [-1.874980194, 18.54625834, 0] 
realBottom = [-10.125, 16.5, 0]

realLeft = [-12.773099, 27.17649622, 0]
realTop = [-4.523079224, 29.22275456, 0]

#put real coordinates in array
realp[0] = realLeft
realp[1] = realTop
realp[2] = realRight
realp[3] = realBottom



#convertImg = cv.imread('paperCalibrate.jpg')
#cv.imshow("original", convertImg)

#use the undistored image (from intrinsics)
#convertImg = cv.undistort(convertImg, mtx, dist, None, newcameramtx)

#hsv = cv.cvtColor(convertImg, cv.COLOR_BGR2HSV)

#hsv boundaries
#lowerWhite = np.array([0, 0, 100])
#upperWhite = np.array([90, 90, 255])   

#perform hsv mask on image
#mask = cv.inRange(hsv, lowerWhite, upperWhite)
#maskx = cv.inRange(hsv, upperWhite, lowerWhite)

#find contours in the mask
#cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
#cnts = cnts[0] if imutils.is_cv2() else cnts[1]

#only proceed if at least one contour was found
#if len(cnts) > 0:
    # find the largest contour in the mask
    #c = max(cnts, key=cv.contourArea)

    # finding the extreme values
    #extLeft = tuple(c[c[:, :, 0].argmin()][0])
    #extRight = tuple(c[c[:, :, 0].argmax()][0])
    #extTop = tuple(c[c[:, :, 1].argmin()][0])
    #extBot = tuple(c[c[:, :, 1].argmax()][0])

    #cv.drawContours(maskx, [c], -1, (255,0,0), 3)
    #cv.imshow("c", maskx)
    #cv.waitKey(200)


#undistortd points for picture of paper (found manually)
#extLeft = (88, 319)
#extTop = (235, 307)
#extRight = (266, 415)
#extBot = (49, 444)

extLeft = (112, 220)
extTop = (242, 170)
extRight = (268, 418)
extBot = (43, 450)


#create a numpy 4x2 array of zeros in order to store the corresponding pixel point locations
pxlp = np.zeros((4, 2), np.float32)

#put values into array
pxlp[0] = extLeft
pxlp[1] = extTop
pxlp[2] = extRight
pxlp[3] = extBot


#use solve pnp to find extrinsics
retval, rvec, tvec = cv.solvePnP(realp, pxlp, mtx, dist)

#change the 3x1 rotation vector into a 3x3 matrix
rotM = cv.Rodrigues(rvec)[0]

#calculate the camera position
#should be [0, 0, h]
cameraPosition = -np.matrix(rotM).T * np.matrix(tvec)
print(cameraPosition)




#saves intrinsics and extrinsics to file
np.savez('cam0M.npz', mtx=mtx, dist=dist, newcameramtx=newcameramtx, roi=roi, rotM=rotM, tvec=tvec, cameraPosition=cameraPosition, pxlp=pxlp, realp=realp)


while True:
    key = cv.waitKey(25)
    if key == 27:
        break


cv.destroyAllWindows()

