from collections import deque
import numpy as np
import argparse
import imutils
import cv2

camera = cv2.VideoCapture(0)

while True:

	(grabbed, frame) = camera.read()
	#frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)

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
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

	#detect canny edges
	#paramters: source, threshold1, threshold2
	edges = cv2.Canny(mask, 75, 150)

	# find contours in the mask and initialize the current
        # (x, y) center of the ball
        #cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        center = None

	go = False

        # only proceed if at least one contour was found
        if len(cnts) > 0:
        	# find the largest contour in the mask, then use
        	# it to compute the minimum enclosing circle and
        	# centroid
        	c = max(cnts, key=cv2.contourArea)
        	x, y, w, h = cv2.boundingRect(c)
		((a, b), radius) = cv2.minEnclosingCircle(c)
		M = cv2.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
		

        	# finding the extreme values
        	extLeft = tuple(c[c[:, :, 0].argmin()][0])
        	extRight = tuple(c[c[:, :, 0].argmax()][0])
        	extTop = tuple(c[c[:, :, 1].argmin()][0])
        	extBot = tuple(c[c[:, :, 1].argmax()][0])

		cv2.circle(frame, extLeft, 5, (0, 0, 255), -1)  #red left point
		cv2.circle(frame, extRight, 5, (0, 255, 0), -1)  #green right point
		cv2.circle(frame, extTop, 5, (255, 0, 0), -1)  #blue top point
		cv2.circle(frame, extBot, 5, (255, 255, 0), -1)  #teal bottom point

		cv2.circle(frame, (170, 143), 5, (0, 0, 255), -1)  #red left point
		cv2.circle(frame, (318, 228), 5, (0, 255, 0), -1)  #green right point
		cv2.circle(frame, (261, 105), 5, (255, 0, 0), -1)  #blue top point
		cv2.circle(frame, (233, 263), 5, (255, 255, 0), -1)  #teal bottom point

		print "left"
		print extLeft
		print "right"
		print extRight
		print "top"
		print extTop
		print "bot"
		print extBot




	cv2.imshow("frame", frame)
	cv2.imshow("mask", mask)
	key = cv2.waitKey(1) & 0xFF

	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
