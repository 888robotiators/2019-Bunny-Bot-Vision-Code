#!/usr/bin/python
#print "looking"
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

degree_angle = None
dist         = npzFile['dist']
increment    = 0
message      = None
mtx          = npzFile['mtx']
newcameramtx = npzFile['newcameramtx']
npzFile      = np.load('cam01.npz')
otherAddress = None
#pts         = np.array([(172, 218), (404, 212), (525, 325), (41, 334)], dtype = "float32")
pts          = np.array([(363, 176), (574, 198), (515, 430), (111, 310)], dtype = "float32")
realHeight   = 33
#realHeight  = 28.5
realWidth    = 18.5
#realWidth   = 27.5
rioAddress   = ('10.8.88.2', 5805)
roi          = npzFile['roi']
scalar       = 16
scalarHeight = int(realHeight * scalar)
scalarWidth  = int(realWidth * scalar)
video        = cv2.VideoCapture(0)

dst          = np.array([[0, 0],
                         [scalarWidth - 1, 0],
                         [scalarWidth - 1, scalarHeight - 1],
                         [0, scalarHeight - 1]],
                         dtype = "float32")
M            = cv2.getPerspectiveTransform(pts, dst)

while True:

    #prepare the frame
    #start capturing images from the video
    grabbed, frame = video.read()

    if not grabbed:
        continue

    frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)

    warped = cv2.warpPerspective(frame, M, (scalarWidth, scalarHeight))
    cv2.circle(frame, (363, 176), 5, (100, 255, 255), -1)  #yellow first point
    cv2.circle(frame, (111, 310), 5, (0, 0, 255), -1)  #red last point
    cv2.circle(frame, (515, 430), 5, (0, 255, 0), -1)  #green third point
    cv2.circle(frame, (574, 196), 5, (255, 0, 0), -1)  #blue second point

    #perform a gaussian blur on image
    #parameters: source, (sigmaX, sigmaY), border type (0 = defualt)

    #convert from bgr to hsv color space
    #hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    #convert from bgr to gray color space

    lowerWhite = np.array([0, 0, 220])
    upperWhite = np.array([60, 200, 255])

    #lowerWhite = np.array([0, 0, 180])
    #upperWhite = np.array([160, 235, 255])
    #perform hsv mask on image
    mask = cv2.inRange(hsv, lowerWhite, upperWhite)
    mask2 = mask.copy()
    contours, heirarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    #cv2.imshow("image", frame)
    #cv2.imshow("mask", mask2)
    #cv2.imshow("hsv", hsv)
    if len(contours) > 0:
        # find the largest contour in the mask
        c = max(contours, key=cv2.contourArea)

        rect = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(warped,[box],0,(0,0,255),2)
        #print("Box")
        #print(box)

        #realTopLeft = (-16.5, 57)
        realTopLeft = (-9.25, 42)
        floatScale = float(scalar)
        pixelToInch = float(1 / floatScale)
        #trueInch = (((truePixel[0] * pixelToInch) + realTopLeft[0]), (realTopLeft[1] - (truePixel[1] * pixelToInch))) #currently used
        dist0 = math.sqrt((box[0][0] - box[1][0]) ** 2 + (box[0][1] - box[1][1]) ** 2)
        dist1 = math.sqrt((box[1][0] - box[2][0]) ** 2 + (box[1][1] - box[2][1]) ** 2)
        if (dist0 < dist1):
            temp1 = ((box[0][0] + box[1][0]) / 2, (box[0][1] + box[1][1]) / 2)
            temp2 = ((box[2][0] + box[3][0]) / 2, (box[2][1] + box[3][1]) / 2)
        else:
            temp1 = ((box[0][0] + box[3][0]) / 2, (box[0][1] + box[3][1]) / 2)
            temp2 = ((box[1][0] + box[2][0]) / 2, (box[1][1] + box[2][1]) / 2)
        #cv2.line(warped, temp1, temp2, (0, 255, 0), 2)
        #print temp1
        #print temp2
        #print dist0
        #print dist1
        if (temp1[1] > temp2[1]):
            target = temp1
            #angle = math.atan2((temp2[0] - temp1[0]), (temp2[1] - temp1[1]))
        else:
            target = temp2
            #angle = math.atan2((temp1[1] - temp2[1]), (temp1[0] - temp2[0]))

        angle = math.atan2((temp2[0] - temp1[0]), (temp1[1] - temp2[1]))
        if angle > (np.pi / 2):
            angle -= np.pi
        #print "angle"
        print (angle * 180 / math.pi)
        angle = angle * 180 / math.pi
        target = (((target[0] * pixelToInch) + realTopLeft[0]), (realTopLeft[1] - (target[1] * pixelToInch))) #currently used
        print target
        #print float(increment), float(target[0]), float(target[1]), float(angle)
        #cv2.circle(warped, temp1, 7, (0, 166, 255), -1)  #yellow first point
        #cv2.circle(warped, temp2, 7, (0, 166, 255), -1)  #yellow first point
        #cv2.circle(warped, tuple(box[0]), 7, (0, 255, 0), -1)
        #cv2.circle(warped, tuple(box[1]), 7, (255, 0, 0), -1)
        #cv2.circle(warped, tuple(box[2]), 7, (0, 0, 255), -1)
        #cv2.circle(warped, tuple(box[3]), 7, (255, 165, 0), -1)
        #cv2.circle(warped, target, 7, (0, 0, 0), -1)  #yellow first point
        if otherAddress is not None:
            if target is not None:
                send = struct.pack('!ffff', float(increment), float(target[0]), float(target[1]), float(angle))
                #print "length"
                #print len(send)
                #print send
                sock.sendto(send, rioAddress)
                increment += 1



    elif otherAddress is not None:
        send = struct.pack('!ffff', -1.0, -1.0, -1.0, -1.0)
        #print "length"
        #print len(send)
        #print rioAddress
        sock.sendto(send, rioAddress)
    #cv2.imshow("warped", warped)
    key = cv2.waitKey(25)
    if key == 27:
        break
video.release()
cv2.destroyAllWindows()

    #print "Looking"
    #while otherAddress is None:
    #    try:

    #        cycle, otherAddress = sock.recvfrom(65507)
    #        print cycle
    #        print otherAddress

    #    except Exception:
    #        cycle = None
    #        rioAddress = None
