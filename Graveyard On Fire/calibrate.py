import numpy as np
import cv2 as cv
import glob
from collections import deque
import argparse
import imutils

#start the video feed
video = cv.VideoCapture(0)

#termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

#prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((9*7, 3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:7].T.reshape(-1,2)

# Arrays to store object points and image points from all the images
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image space

i = 0

while True:
	(grabbed, img) = video.read()
	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	#cv.imshow("Image", img)
	#cv.imshow("Gray", gray)

	# Find the chess board corners
	ret, corners = cv.findChessboardCorners(gray, (9,7), None)
	
	# If found, add object points, image points (after refining them)
	if ret == True:
		objpoints.append(objp)
		
		corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
		imgpoints.append(corners)
		
		# Draw and dispay the corners
		cv.drawChessboardCorners(img, (9,7), corners2, ret)
		cv.imshow('img', img)
		cv.imwrite('pic{:>05}.jpg'.format(i), img)
		if cv.WaitKey(10) == 27:
			break
		i += 1
		cv.waitKey(5)

video.release()
cv.destroyAllWindows()
