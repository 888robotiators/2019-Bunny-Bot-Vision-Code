import cv2
import numpy as np
import imutils

npzFile = np.load('cam01.npz')
dist = npzFile['dist']
mtx = npzFile['mtx']
newcameramtx = npzFile['newcameramtx']

img = cv2.imread('paperCalibrate2.jpg')

#cv2.imshow("original", img)

img2 = img.copy()


img2 = cv2.undistort(img, mtx, dist, None, 
mtx)
img = cv2.undistort(img, mtx, dist, None, mtx)
#cv2.imshow("undistort", img2)

#perform a gaussian blur on image
#parameters: source, (sigmaX, sigmaY), border type (0 = defualt)
#blurred_frame = cv2.GaussianBlur(img2, (5,5), 0)
    
#convert from bgr to hsv color space
#hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)
hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

lowerWhite = np.array([0, 0, 140])
upperWhite = np.array([100, 120, 255])   

#perform hsv mask on image
mask = cv2.inRange(hsv, lowerWhite, upperWhite)

cv2.imshow("hsv", mask)

mask = cv2.erode(mask, None, iterations=1)
mask = cv2.dilate(mask, None, iterations=1)

cv2.imshow("better", mask)

# find contours in the mask

#cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]

# only proceed if at least one contour was found
if len(cnts) > 0:
      	# find the largest contour in the mask
        c = max(cnts, key=cv2.contourArea)

	cnts_ordered = sorted(cnts, key=cv2.contourArea)
	c2 = cnts_ordered[len(cnts_ordered) - 2]
	
	cv2.drawContours(img2, cnts, -1, (0,255,0), 3)
	cv2.drawContours(img2, c, -1, (0,0,255), 3)
	cv2.drawContours(img2, c2, -1, (255,0,0), 3)
	cv2.imshow("contour", img2)

        # finding the extreme values
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])

	#extLeft = (88, 319)
	#extTop = (235, 307)
	#extRight = (266, 420)
	#extBot = (49, 444)
	#extPoint = (245, 175)
	#extPoint2 = (115, 220)

	extLeft = (83, 323)
	extTop = (237, 307)
	extRight = (268, 418)
	extBot = (43, 450)
	extPoint = (242, 170)
	extPoint2 = (112, 220)
	extPoint3 = (275, 120)
	extPoint4 = (425, 160)
	extPoint5 = (88, 292)

	cv2.circle(img, extLeft, 2, (0, 0, 255), -1)  #red left point
	cv2.circle(img, extRight, 2, (0, 255, 0), -1)  #green right point
	cv2.circle(img, extTop, 2, (255, 0, 0), -1)  #blue top point
	cv2.circle(img, extBot, 2, (255, 255, 0), -1)  #teal bottom point

	cv2.circle(img, extPoint, 2, (255, 255, 0), -1)  #teal higher tape point
	cv2.circle(img, extPoint2, 2, (255, 255, 0), -1)  #teal lower tape point
	cv2.circle(img, extPoint3, 2, (255, 255, 0), -1)  #teal door point
	cv2.circle(img, extPoint4, 2, (255, 255, 0), -1)  #teal mid tape point
	cv2.circle(img, extPoint5, 2, (255, 255, 0), -1)  #teal black paper point
	
	cv2.imshow("points", img)

	print "extreme left"
	print extLeft
	print "extreme right"
	print extRight
	print "extreme top"
	print extTop
	print "extreme bottom"
	print extBot

while True:
	key = cv2.waitKey(25)
	if key == 27:
    	    break

cv2.destroyAllWindows()
