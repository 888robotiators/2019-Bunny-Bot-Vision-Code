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
	# show the frame to our screen
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	cv2.imshow("frame", frame)
        cv2.imshow("HSV", hsv)
	key = cv2.waitKey(1) & 0xFF

	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
