#!/usr/bin/python3

import cv2
import numpy as np
import math
import struct
import socket
import sys
import time
import select

video = cv2.VideoCapture(0)

npzFile = np.load('cam01.npz')
dist = npzFile['dist']
mtx = npzFile['mtx']
newcameramtx = npzFile['newcameramtx']
roi = npzFile['roi']
rotM = npzFile['rotM']
tvec = npzFile['tvec']
cameraPosition = npzFile['cameraPosition']
pxlp = npzFile['pxlp']
realp = npzFile['realp']

z = 0.0
camMat = np.asarray(mtx)
iRot = np.linalg.inv(rotM)
iCam = np.linalg.inv(camMat)

while True:
#prepare the frame
    #start capturing images from the video
    grabbed, distFrame = video.read()

    frame = cv2.undistort(cv2.imread("staticLine.jpg"), mtx, dist, None, newcameramtx)

    if not grabbed:
        print "error"

    #perform a gaussian blur on image
    #parameters: source, (sigmaX, sigmaY), border type (0 = defualt)
    blurred_frame = cv2.GaussianBlur(frame, (5,5), 0)
    
    #convert from bgr to hsv color space
    hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)

    #convert from bgr to gray color space
    gray = cv2.cvtColor(blurred_frame,  cv2.COLOR_BGR2GRAY)

    #define lower and upper bounds of hsv mask to detect white
    #lowerWhite = np.array([45, 20, 165])
    #upperWhite = np.array([140, 120, 255])

    lowerWhite = np.array([0, 0, 190])
    upperWhite = np.array([150, 200, 255])   

    #perform hsv mask on image
    mask = cv2.inRange(hsv, lowerWhite, upperWhite)

    #detect canny edges
    #paramters: source, threshold1, threshold2
    edges = cv2.Canny(mask, 75, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, maxLineGap=50)
    

    if lines is not None:
        print "number of lines"
        print len(lines[0])
        print lines[0]
	
        points = []
        for line in lines[0]:
	    
	    #pixel points of line
            x1, y1, x2, y2 = line
	
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (x1, y1), 5, (0, 255, 255), -1)  #yellow first point
	    cv2.circle(frame, (x2, y2), 5, (0, 0, 255), -1)  #red second point

	    linePixels = [[x1, y1], [x2, y2]]


	    #convert the key pixel points to inches from the camera
	    uvPoint1 = np.ones((3, 1))
	    uvPoint2 = np.ones((3, 1))

	    uvPoint1[0, 0] = x1
	    uvPoint1[1, 0] = y1

	    uvPoint2[0, 0] = x2
	    uvPoint2[1, 0] = y2

	    tempMat = np.matmul(np.matmul(iRot, iCam), uvPoint1)
	    tempMat2 = np.matmul(iRot, tvec)

	    s = (z + tempMat2[2, 0]) / tempMat[2, 0]
	    wcPoint1 = np.matmul(iRot, (np.matmul(s * iCam, uvPoint1) - tvec))

	    # wcPoint[2] will not be exactly equal to z, but very close to it
	    assert int(abs(wcPoint1[2] - z) * (10 ** 8)) == 0
	    wcPoint1[2] = z

	    point1 = wcPoint1.T[0]
	    #print point1

	    tempMat3 = np.matmul(np.matmul(iRot, iCam), uvPoint2)
	    tempMat4 = np.matmul(iRot, tvec)

	    s2 = (z + tempMat4[2, 0]) / tempMat3[2, 0]
	    wcPoint2 = np.matmul(iRot, (np.matmul(s2 * iCam, uvPoint2) - tvec))

	    # wcPoint[2] will not be exactly equal to z, but very close to it
	    assert int(abs(wcPoint2[2] - z) * (10 ** 8)) == 0
	    wcPoint2[2] = z

	    point2 = wcPoint2.T[0]
	    #print point2



            radian_angle = math.atan2((x2-x1),(y1-y2))
            degree_angle = math.degrees(radian_angle)

            if degree_angle > 90:
                degree_angle = degree_angle - 180
            points.append(point1)
            points.append(point2)
    #print degree_angle




    cv2.imshow("frame", frame)
    cv2.imshow("hsv mask", mask)
    cv2.imshow("edges", edges)


    key = cv2.waitKey(25)
    if key == 27:
        break
video.release()
cv2.destroyAllWindows()
