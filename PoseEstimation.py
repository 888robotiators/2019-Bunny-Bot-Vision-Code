# import the necessary packages
from collections import deque
import numpy as np
import argparse
import imutils
import cv2
npzFile = np.load('cam0M.npz')
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
        h, w = frame.shape[:2]
        frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)

        lowerRed = np.array([0,0, 170])
        upperRed = np.array([130,100, 255])
        frame2 = cv2.GaussianBlur(frame2, (3,3), 0)
        redMask = cv2.inRange(frame2, lowerRed, upperRed)

        #def onmouse(k, x, y, s, p):
            #global frame
            #if k==1:
                #print frame[y,x]
        #cv2.namedWindow("frame")
        #cv2.setMouseCallback("frame",onmouse)
        contours, heirarchy = cv2.findContours(redMask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
     	objP = np.array([(-4, 2.5, 0), (-4, -2.5, 0),(-2, 2.5, 0), (-2, -2.5, 0),(2, 2.5, 0) , (2, -2.5, 0), (4, 2.5, 0), (4, -2.5, 0)], dtype = np.float32())
        if len(contours) > 0:
                allApprox = None
                for c in contours:
                        epsilon = 0.1*cv2.arcLength(c,True)
                        approx = cv2.approxPolyDP(c,epsilon,True)
                        #print approx
                        if allApprox == None:
                                allApprox = approx.copy()
                        else:
                                allApprox = np.concatenate((allApprox, approx), axis=0)
                if len(allApprox) == 8:
                        sortApprox = sorted(allApprox, key=lambda k: [k[:,0], k[:,1]])
                        sortApprox = np.array(sortApprox)
                        for i in range(len(sortApprox[:,0])):
                                if i % 2 == 0:
                                        approx1 = sortApprox[i][0].copy()
                                        approx2 = sortApprox[i+1][0].copy()
                                        if (approx1[1] < approx2[1]):
                                                sortApprox[i][0] = approx2.copy()
                                                sortApprox[i+1][0] = approx1.copy()
                        sortApprox = np.array(sortApprox)
                        cv2.drawContours(frame, [sortApprox], 0, (0, 255, 0), 1)
                        sortApprox = sortApprox.astype(np.float32)
                        #print sortApprox
                        retval, rvec, tvec = cv2.solvePnP(objP, sortApprox, mtx, None)
                        #print rvec
                        #print ""
                        #print tvec
                        #print ""
                        inches = tvec[2]
                        cv2.putText(frame, "%.2fft" % (inches / 12), (frame.shape[1]-300, frame.shape[0]-20), cv2.FONT_HERSHEY_TRIPLEX, 2.0, (0, 255, 0), 3)
        
        cv2.imshow("frame", frame)
        #cv2.imshow("frame2", frame2)
        
        cv2.imshow("RedMask", redMask)
	key = cv2.waitKey(1) & 0xFF

	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
