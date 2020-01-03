#!/usr/bin/python

import cv2
import numpy as np
import math
import struct
import socket
import sys
import time
import select

#defines udp server address for jetson
HOST = '10.8.88.19'
PORT = 5809

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((HOST, PORT))
print sock

npzFile = np.load('cam01.npz')
dist = npzFile['dist']
mtx = npzFile['mtx']
newcameramtx = npzFile['newcameramtx']
roi = npzFile['roi'] 

message = None
rioAddress = ('10.8.88.2', 5806)

otherAddress = None

degree_angle = None

#define video of white lines
#change to camera later
capturedVideo = "C:\Users\emily\ROBOT\vision2019\test2.mp4"
video = cv2.VideoCapture(0)
#frame = cv2.imread('white_line_pic.png')
#new_frame = frame.copy()

while True:

    while otherAddress is None:
        try:
	    print "looking"
            cycle, otherAddress = sock.recvfrom(65507)
	    print cycle

        except Exception:
            cycle = None
            rioAddress = None
        
    #start capturing images from the video
    grabbed, frame = video.read()
    frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)
    print type(frame)
    #if video ends, restart
    if not grabbed:
        print "error"
        video = cv2.VideoCapture("white_line_video.mp4")

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
    avgLine = 0

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #cv2.circle(new_frame, (x1, y1), 5, (0, 255, 255), -1)  #yellow first point
	    #cv2.circle(new_frame, (x2, y2), 5, (0, 0, 255), -1)  #red second point
            radian_angle = math.atan2((x2-x1),(y1-y2))
            degree_angle = math.degrees(radian_angle)

            if degree_angle > 90:
                degree_angle = degree_angle - 180
	    
	    avgLine = avgLine + degree_angle
            print degree_angle
	
	avgLine = avgLine/len(lines)
	print avgLine

    if rioAddress is not None:
	if degree_angle is not None:
		send = struct.pack('!f', avgLine)

    		sock.sendto(send, rioAddress)

	else:
		print "oof"

    cv2.imshow("frame", frame)
    #cv2.imshow("blurry boi", blurred_frame)
    #cv2.imshow("hsv thing", hsv)
    cv2.imshow("hsv mask", mask)
    #cv2.imshow("edges", edges)
    #cv2.imshow("new frame", new_frame)

    key = cv2.waitKey(25)
    if key == 27:
        break
video.release()
cv2.destroyAllWindows()
