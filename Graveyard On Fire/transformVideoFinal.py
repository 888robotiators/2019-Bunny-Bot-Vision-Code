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
img = cv2.imread("transformFinal.jpg")
#pts = np.array([(104, 193), (383, 197), (626, 340), (8, 388)], dtype = "float32")
#pts = np.array([(192, 146), (466, 145), (626, 240), (41, 242)], dtype = "float32") 
pts = np.array([(172, 218), (404, 212), (525, 325), (41, 334)], dtype = "float32") 
width = int(math.sqrt(((388-340) ** 2) + ((626-8) ** 2)))
height = int(math.sqrt(((340-193) ** 2) + ((626-383) ** 2)))
#realWidth = 39.75 # must be changed whenever camera is moved
#realHeight = 32 # must be changed whenever camera is used
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
    
    
    
    
    warped = cv2.warpPerspective(frame, M, (scalarWidth, scalarHeight))

    
    cv2.circle(img, (104, 193), 3, (100, 255, 255), -1)  #yellow first point
    cv2.circle(img, (8, 388), 3, (0, 0, 255), -1)  #red last point
    cv2.circle(img, (626, 340), 3, (0, 255, 0), -1)  #green third point
    cv2.circle(img, (383, 197), 3, (255, 0, 0), -1)  #blue second point
    
    """
    cv2.circle(frame, (192, 146), 3, (100, 255, 255), -1)  #yellow first point
    cv2.circle(frame, (41, 242), 3, (0, 0, 255), -1)  #red last point
    cv2.circle(frame, (626, 240), 3, (0, 255, 0), -1)  #green third point
    cv2.circle(frame, (466, 145), 3, (255, 0, 0), -1)  #blue second point
    """
    cv2.circle(frame, (172, 218), 5, (100, 255, 255), -1)  #yellow first point
    cv2.circle(frame, (41, 334), 5, (0, 0, 255), -1)  #red last point
    cv2.circle(frame, (525, 325), 5, (0, 255, 0), -1)  #green third point
    cv2.circle(frame, (404, 212), 5, (255, 0, 0), -1)  #blue second point
    
    
    #print pts
    

    #perform a gaussian blur on image
    #parameters: source, (sigmaX, sigmaY), border type (0 = defualt)
    #blurred_frame = cv2.GaussianBlur(warped, (5,5), 0)

    #convert from bgr to hsv color space
    #hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    #convert from bgr to gray color space
    #gray = cv2.cvtColor(blurred_frame,  cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(warped,  cv2.COLOR_BGR2GRAY)
    #define lower and upper bounds of hsv mask to detect white
    #lowerWhite = np.array([45, 20, 165])
    #upperWhite = np.array([140, 120, 255])

    lowerWhite = np.array([0, 0, 200])
    upperWhite = np.array([100, 200, 255])   

    #lowerWhite = np.array([0, 0, 180])
    #upperWhite = np.array([160, 235, 255]) 
    #perform hsv mask on image
    mask = cv2.inRange(hsv, lowerWhite, upperWhite)
    mask2 = mask.copy()
    gray2 = gray.copy()
    contours, heirarchy = cv2.findContours(mask2, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    if len(contours) > 0:

        cnt = contours[0]

        rect = cv2.minAreaRect(cnt)
        box = cv2.cv.BoxPoints(rect)
        box = np.int0(box)
        #cv2.drawContours(warped,[box],0,(0,0,255),2)
        #print("Box")
        #print(box)
    if len(contours) > 0:
        # find the largest contour in the mask
        c = max(contours, key=cv2.contourArea)

        # finding the extreme values
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])
        #cv2.drawContours(warped, [c], -1, (255,0,0), 3)
        #cv2.drawContours(hsv, [c], -1, (255,0,0), 3)
    #detect canny edges
    #paramters: source, threshold1, threshold2
    edges = cv2.Canny(mask, 75, 150)
    #warped = four_point_transform(img, pts)
    
    #cv2.imshow("image", frame)
    #cv2.imshow("mask", mask)
    #cv2.imshow("contours", mask2)
    #cv2.imshow("gray", gray)
    #cv2.imshow("edges", edges)
    #cv2.imshow("hsv", hsv)
    
    retval = cv2.moments(mask2, binaryImage = True)
    hu = cv2.HuMoments(retval)
    #print retval
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, maxLineGap=50)
    avgLine = 0
    numLines = 0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            #print line
            cv2.line(warped, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(warped, (x1, y1), 5, (0, 255, 255), -1)  #yellow first point
        
            cv2.circle(warped, (x2, y2), 5, (0, 0, 255), -1)  #red second point
            radian_angle = math.atan2((x2-x1),(y1-y2))
            #degree_angle = math.degrees(radian_angle)
            if radian_angle > (np.pi / 4):
                radian_angle = radian_angle - (np.pi)
            #if degree_angle > 90:
                #degree_angle = degree_angle - 180
            avgLine = avgLine + radian_angle
            #print degree_angle
            numLines += 1
	
        avgLine = avgLine/len(lines)
        avgLine = math.degrees(avgLine)
        #print hu[0][0] # I think this is moment of inertia
        diameter = math.sqrt(((extBot[0] - extTop[0]) **2) + ((extBot[1] - extTop[1]) ** 2))
        #realTopLeft = (-9.25, 89.75)#Must be changed whenever the camera is moved
        realTopLeft = (-16.5, 57)
        floatScale = float(scalar)
        pixelToInch = float(1 / floatScale)
        # Formula for converting from pixel to inch:
        pixel = (((extBot[0] + extTop[0])/2), ((extBot[1] + extTop[1]) / 2)) #Not currently used
        topPixel = (extTop[0], extTop[1]) #not currently used
        #print "pixel"
        #print pixel
        #print "scalar"
        #print scalar
        #print "pixelToInch"
        #print pixelToInch
        inch = (((pixel[0] * pixelToInch) + realTopLeft[0]), (realTopLeft[1] - (pixel[1] * pixelToInch)))
        #finds the point in the middle of the bottom of the line
        if avgLine > 2 and avgLine < -2:
            truePixel = extBot
        elif avgLine > 2:
            truePixel = (((extBot[0] + extLeft[0]) / 2), ((extBot[1] + extLeft[1]) / 2))
        else:
            truePixel = (((extBot[0] + extRight[0]) / 2), ((extBot[1] + extRight[1]) / 2))

	#for testing
        truePixel = extBot
	#print "pixel"
	#print truePixel
        
        trueInch = (((truePixel[0] * pixelToInch) + realTopLeft[0]), (realTopLeft[1] - (truePixel[1] * pixelToInch))) #currently used
        topInch = (((topPixel[0] * pixelToInch) + realTopLeft[0]), (realTopLeft[1] - (topPixel[1] * pixelToInch)))
	
	#print "truePixel[0] * pixelToInch"
	#print truePixel[0] * pixelToInch
	#print truePixel[0]
        print pixelToInch

        #heading = math.atan2(inch[0] / inch[1])
        trueHeading = math.atan2(trueInch[0] , trueInch[1])
        #math.radians(avgLine)
        #math.radians(heading)
        
        print "inch"
        print trueInch
        #print "heading"
        #print heading
        print "lineAngle"
        print avgLine
        print "Lines length"
        print len(lines)
        print "NumLines"
        print numLines
        cv2.circle(warped, truePixel, 7, (100, 255, 255), -1)  #yellow first point
        cv2.circle(warped, extBot, 10, (255, 0, 0), -1)  #yellow first point
        cv2.circle(warped, extRight, 5, (0, 255, 0), -1)  #yellow first point
        cv2.circle(warped, extLeft, 5, (0, 0, 255), -1)  #yellow first point

        if otherAddress is not None:
            if trueInch is not None and avgLine is not None:
                send = struct.pack('!ffff', increment, trueInch[0], trueInch[1], avgLine)
                #print "length"
                #print len(send)
                sock.sendto(send, rioAddress)
                increment += 1

            
                
        else:
            print "oof"

    elif otherAddress is not None:
        send = struct.pack('!ffff', -1, -1, -1, -1)
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
