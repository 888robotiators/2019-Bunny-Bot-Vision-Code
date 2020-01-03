#!/usr/bin/python
print "looking"
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


npzFile = np.load('cam04.npz')
dist = npzFile['dist']
mtx = npzFile['mtx']
newcameramtx = npzFile['newcameramtx']
roi = npzFile['roi']

message = None
rioAddress = ('10.8.88.2', 5805)

otherAddress = None

degree_angle = None

video = cv2.VideoCapture(0)
pts = np.array([(172, 218), (404, 212), (525, 325), (41, 334)], dtype = "float32") 
realHeight = 28.5
realWidth = 27.5
scalar = 16
scalarWidth = int(realWidth * scalar)
scalarHeight = int(realHeight * scalar)
dst = np.array([
    [0, 0],
    [scalarWidth - 1, 0],
    [scalarWidth - 1, scalarHeight - 1],
    [0, scalarHeight - 1]], dtype = "float32")
M = cv2.getPerspectiveTransform(pts, dst)
increment = 0

while True:
#prepare the frame
    #start capturing images from the video
    grabbed, frame = video.read()
    frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)
    
    """
    print "Looking"
    while otherAddress is None:
        try:
           
            cycle, otherAddress = sock.recvfrom(65507)
            print cycle
            print otherAddress

        except Exception:
            cycle = None
            rioAddress = None
    if not grabbed:
        print "error"
    """
    
    
    
    warped = cv2.warpPerspective(frame, M, (scalarWidth, scalarHeight))
    cv2.circle(frame, (172, 218), 5, (100, 255, 255), -1)  #yellow first point
    cv2.circle(frame, (41, 334), 5, (0, 0, 255), -1)  #red last point
    cv2.circle(frame, (525, 325), 5, (0, 255, 0), -1)  #green third point
    cv2.circle(frame, (404, 212), 5, (255, 0, 0), -1)  #blue second point
    

    #perform a gaussian blur on image
    #parameters: source, (sigmaX, sigmaY), border type (0 = defualt)

    #convert from bgr to hsv color space
    #hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    #convert from bgr to gray color space

    lowerWhite = np.array([0, 0, 200])
    upperWhite = np.array([100, 200, 255])   

    #lowerWhite = np.array([0, 0, 180])
    #upperWhite = np.array([160, 235, 255]) 
    #perform hsv mask on image
    mask = cv2.inRange(hsv, lowerWhite, upperWhite)
    mask2 = mask.copy()
    contours, heirarchy = cv2.findContours(mask2, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        # find the largest contour in the mask
        c = max(contours, key=cv2.contourArea)

        # finding the extreme values
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])

        rect = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(warped,[box],0,(0,0,255),2)
        print("Box")
        print(box)
    #detect canny edges
    #paramters: source, threshold1, threshold2
    edges = cv2.Canny(mask, 75, 150)
    
    cv2.imshow("image", frame)
    #cv2.imshow("mask", mask)
    cv2.imshow("contours", mask2)
    #cv2.imshow("gray", gray)
    #cv2.imshow("edges", edges)
    #cv2.imshow("hsv", hsv)
    retval = cv2.moments(mask2, binaryImage = True)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, maxLineGap=50)
    avgLine = 0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            radian_angle = math.atan2((x2-x1),(y1-y2))
            #degree_angle = math.degrees(radian_angle)
            if radian_angle > (np.pi / 4):
                radian_angle = radian_angle - (np.pi)
            #if degree_angle > 90:
                #degree_angle = degree_angle - 180
            avgLine = avgLine + radian_angle
            #print degree_angle
	    #cv2.line(warped, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #cv2.circle(warped, (x1, y1), 5, (0, 255, 255), -1)  #yellow first point
        
            #cv2.circle(warped, (x2, y2), 5, (0, 0, 255), -1)  #red second point
        avgLine = avgLine/len(lines)
        avgLine = math.degrees(avgLine)

        realTopLeft = (-16.5, 57)
        floatScale = float(scalar)
        pixelToInch = float(1 / floatScale)
        temp = ((box[0][0] + box[1][0]) / 2, (box[0][1] + box[1][1]) / 2)
	#for testing
        truePixel = extBot
	print "pixel"
	print truePixel[0]
        print type(truePixel[0])
        trueInch = (((truePixel[0] * pixelToInch) + realTopLeft[0]), (realTopLeft[1] - (truePixel[1] * pixelToInch))) #currently used
        #box0 = (((box[0][0] * pixelToInch) + realTopLeft[0]), (box[0][1] - (truePixel[1] * pixelToInch))) #currently used
        #box1 = (((box[1][0] * pixelToInch) + realTopLeft[0]), (box[1][1] - (truePixel[1] * pixelToInch))) #currently used
        #box2 = (((box[2][0] * pixelToInch) + realTopLeft[0]), (box[2][1] - (truePixel[1] * pixelToInch))) #currently used
        #box3 = (((box[3][0] * pixelToInch) + realTopLeft[0]), (box[3][1] - (truePixel[1] * pixelToInch))) #currently used
        dist0 = math.sqrt((box[0][0] - box[1][0]) ** 2 + (box[0][1] - box[1][1]) ** 2)
        dist1 = math.sqrt((box[1][0] - box[2][0]) ** 2 + (box[1][1] - box[2][1]) ** 2)
        if (dist0 < dist1):
            temp1 = ((box[0][0] + box[1][0]) / 2, (box[0][1] + box[1][1]) / 2)
            temp2 = ((box[2][0] + box[3][0]) / 2, (box[2][1] + box[3][1]) / 2)
        else:
            temp1 = ((box[0][0] + box[3][0]) / 2, (box[0][1] + box[3][1]) / 2)
            temp2 = ((box[1][0] + box[2][0]) / 2, (box[1][1] + box[2][1]) / 2)
        cv2.line(warped, temp1, temp2, (0, 255, 0), 2)
        print temp1
        print temp2
        print dist0
        print dist1
        if (temp1[1] > temp2[1]):
            target = temp1
        else:
            target = temp2
        
        cv2.circle(warped, temp1, 7, (0, 166, 255), -1)  #yellow first point
        cv2.circle(warped, temp2, 7, (0, 166, 255), -1)  #yellow first point
        #cv2.circle(warped, truePixel, 7, (100, 255, 255), -1)  #yellow first point
        #cv2.circle(warped, extBot, 10, (255, 0, 0), -1)  #yellow first point
        #cv2.circle(warped, extRight, 5, (0, 255, 0), -1)  #yellow first point
        #cv2.circle(warped, extLeft, 5, (0, 0, 255), -1)  #yellow first point
        #cv2.circle(warped, temp, 7, (0, 0, 255), -1)  #yellow first point
        cv2.circle(warped, tuple(box[0]), 7, (0, 255, 0), -1)
        cv2.circle(warped, tuple(box[1]), 7, (255, 0, 0), -1)
        cv2.circle(warped, tuple(box[2]), 7, (0, 0, 255), -1)
        cv2.circle(warped, tuple(box[3]), 7, (255, 165, 0), -1)
        cv2.circle(warped, target, 7, (0, 0, 0), -1)  #yellow first point
        if otherAddress is not None:
            if trueInch is not None and avgLine is not None:
                send = struct.pack('!ffff', increment, trueInch[0], trueInch[1], avgLine)
                #print "length"
                #print len(send)
                sock.sendto(send, rioAddress)
                increment += 1
            
                

    elif otherAddress is not None:
        send = struct.pack('!ffff', -1, -1, -1, -1)
        #print "length"
        #print len(send)
        #print rioAddress
        sock.sendto(send, rioAddress)   
    cv2.imshow("warped", warped)
    key = cv2.waitKey(25)
    if key == 27:
        break
video.release()
cv2.destroyAllWindows()
