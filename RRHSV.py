# import the necessary packages
from collections import deque
import numpy as np
import argparse
import imutils
import cv2
npzFile = np.load('cam05.npz')
dist = npzFile['dist']
mtx = npzFile['mtx']
newcameramtx = npzFile['newcameramtx']
roi = npzFile['roi']

camera = cv2.VideoCapture(0)



# keep looping
while True:
	# grab the current frame
	(grabbed, frame) = camera.read()
        frame2 = frame.copy()
        frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        r = frame.copy()
        r[:,:,0] = 0
        r[:,:,1] = 0

        g = frame.copy()
        g[:,:,0] = 0
        g[:,:,2] = 0

        b = frame.copy()
        b[:,:,1] = 0
        b[:,:,2] = 0

        #lowerWhite = np.array([0, 0, 190])
        #upperWhite = np.array([100, 200, 255])   
        lowerWhite = np.array([0, 100, 190])
        upperWhite = np.array([25, 255, 255])
        lowerRed = np.array([0,0, 200])
        upperRed = np.array([100,220, 255])
        redMask = cv2.inRange(r, lowerRed, upperRed)   

        #perform hsv mask on image
        mask = cv2.inRange(hsv, lowerWhite, upperWhite)
	# show the frame to our screen
	gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
        #def onmouse(k, x, y, s, p):
            #global frame
            #if k==1:
                #print frame[y,x]
        #cv2.namedWindow("frame")
        #cv2.setMouseCallback("frame",onmouse)
        contours, heirarchy = cv2.findContours(redMask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        print type(contours)
        print len(contours)
        if len(contours) > 1:
                c = max(contours, key=cv2.contourArea)
                print c
                i = max(contours, key=cv2.contourArea)
                contours.pop(i)
                c2 = max(contours, key=cv2.contourArea)
        #if len(contours) > 0:
                #c = max(contours, key=cv2.contourArea)
                #contours.remove(c)

        #if len(contours) > 0:
                #c2 = max(contours, key=cv2.contourArea)


        bigBox = np.concatenate((c,c2), axis=0)
        #bigBox = c
        rect = cv2.minAreaRect(bigBox)
        box = cv2.cv.BoxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(frame,[box],0,(0,0,255),2)
        #print rect
        distance = 20.5
        focalWidth = 750.9799
        #focalLength = (rect[1][0] * distance) / 8
        inches = (8 *focalWidth) / (rect[1][0])
        cv2.putText(frame, "%.2fft" % (inches / 12), (frame.shape[1]-200, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 3)
	cv2.imshow("frame", frame)
        #cv2.imshow("hsv", hsv)
        cv2.imshow("red", r)
        cv2.imshow("RedMask", redMask)
        #cv2.imshow("green", g)
        #cv2.imshow("blue", b)
        #cv2.imshow("gray", gray)
        #cv2.imshow("hsv mask", mask)
	key = cv2.waitKey(1) & 0xFF

	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
