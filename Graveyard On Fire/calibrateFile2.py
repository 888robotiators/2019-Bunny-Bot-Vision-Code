import numpy as np
import cv2 as cv
import glob
from collections import deque
import argparse
import imutils


# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

#prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((9*7, 3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:7].T.reshape(-1,2)


#to inches
for (i, element) in enumerate(objp):
	objp[i] = element * 20 / 25.4
# Arrays to store object points and image points from all the images
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image space


images = glob.glob('*.jpg')


for fname in images:

	img = cv.imread(fname)
	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	
	#cv.imshow("Image", img)
	#cv.imshow("Gray", gray)
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
img = cv.imread('calibrate001.jpg')
h, w = img.shape[:2]
newcameramtx, roi= cv.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))




#undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)

#crop the image
x,y,w,h = roi








#other code

realp = np.zeros((4, 3), np.float32)
#realp[:,:2] = np.mgrid[0:9,0:7].T.reshape(-1,2)

# Arrays to store object points and image points from all the images
#top left, top right, bottom left, bottom right
realpoints = [] # 3d points of the corners in real world space
pixelpoints = [] # 2d points of the corners in pixel lotation

#inches
realLeft = np.array([-9.5, 24.25, 0])
realTop = np.array([-3.83458, 33.6788, 0])
realRight = np.array([3.415418, 29.4288, 0])
realBottom = np.array([-2.25, 20, 0])

#milimeters
#realLeft = np.array([628.65, 1298.575, 0])
#realTop = np.array([234.95, 1936.75, 0])
#realRight = np.array([676.275, 1320.8, 0])
#realBottom = np.array([114.3, 695.325, 0])

realp[0] = realLeft
realp[1] = realTop
realp[2] = realRight
realp[3] = realBottom



convertImg = cv.imread('paperCalibrate.jpg')

cv.imshow("original", convertImg)

#convertImg = cv.undistort(convertImg, mtx, dist, None, newcameramtx)

#cv.imshow("undistort", convertImg)

#perform a gaussian blur on image
#parameters: source, (sigmaX, sigmaY), border type (0 = defualt)
blurred_frame = cv.GaussianBlur(convertImg, (5,5), 0)
    
#convert from bgr to hsv color space
hsv = cv.cvtColor(blurred_frame, cv.COLOR_BGR2HSV)

lowerWhite = np.array([0, 0, 100])
upperWhite = np.array([90, 90, 255])   

#perform hsv mask on image
mask = cv.inRange(hsv, lowerWhite, upperWhite)


mask = cv.erode(mask, None, iterations=1)
mask = cv.dilate(mask, None, iterations=1)


# find contours in the mask

#cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]

# only proceed if at least one contour was found
if len(cnts) > 0:
      	# find the largest contour in the mask
        c = max(cnts, key=cv.contourArea)
	
        # finding the extreme values
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])

print "extBot"
print extBot

extLeft = (153, 332)
extRight = (383, 300)
extTop = (276, 270)
extBot = (276, 378)

extLeft2 = (152, 334)
extRight2 = (387, 301)
extTop2 = (277, 273)
extBot2 = (278, 382)


pxlp = np.zeros((4, 2), np.float32)
pxlp[0] = extLeft2
pxlp[1] = extTop2
pxlp[2] = extRight2
pxlp[3] = extBot2

"""
pixelpoints.append(extLeft)
pixelpoints.append(extRight)
pixelpoints.append(extTop)
pixelpoints.append(extBot)
"""

retval, rvec, tvec = cv.solvePnP(realp, pxlp, mtx, dist)

rotM = cv.Rodrigues(rvec)[0]

#rvec2, tvec2, inlier = cv.solvePnPRansac(realp, pxlp, mtx, dist)

#rotM2 = cv.Rodrigues(rvec2)[0]



cameraPosition = -np.matrix(rotM).T * np.matrix(tvec)


print "mtx"
print type(mtx)
print len(mtx)
print mtx


print "dist"
print type(dist)
print len(dist)
print dist
print "roi"
print type(roi)
print len(roi)
print roi
print "rotM"
print type(rotM)
print len(rotM)
print rotM
print "tvec"
print type(tvec)
print len(tvec)
print tvec
print "cameraPosition"
print type(cameraPosition)
print len(cameraPosition)
print cameraPosition



tot_error = 0
for i in xrange(len(objpoints)):
	imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
	error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
	tot_error += error

print "mean error: ", tot_error/len(objpoints)
print "total error: ", tot_error
#saves to file
#np.savez('cam01.npz', mtx=mtx, dist=dist, newcameramtx=newcameramtx, roi=roi, rotM=rotM, tvec=tvec, cameraPosition=cameraPosition, pxlp=pxlp)


while True:
	key = cv.waitKey(25)
	if key == 27:
    	    break


cv.destroyAllWindows()
