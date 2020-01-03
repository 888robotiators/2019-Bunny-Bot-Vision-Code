import cv2 as cv
import time
import numpy as np
import argparse
import imutils
from collections import deque
video = cv.VideoCapture(0)


while True:
	(grabbed, img) = video.read()
	cv.imshow('img', img)
	key = cv.waitKey(1) & 0xFF
	if key == ord("1"):
		cv.imwrite('calibrateM01.jpg', img)
	if key == ord("2"):
		cv.imwrite('calibrateM02.jpg', img)
	if key == ord("3"):
		cv.imwrite('calibrateM03.jpg', img)
	if key == ord("4"):
		cv.imwrite('calibrateM04.jpg', img)
	if key == ord("5"):
		cv.imwrite('calibrateM05.jpg', img)
	if key == ord("6"):
		cv.imwrite('calibrateM06.jpg', img)
	if key == ord("7"):
		cv.imwrite('calibrateM07.jpg', img)
	if key == ord("8"):
		cv.imwrite('calibrateM08.jpg', img)
	if key == ord("9"):
		cv.imwrite('calibrateM09.jpg', img)
	if key == ord("0"):
		cv.imwrite('calibrateM00.jpg', img)
	if key == ord("a"):
		cv.imwrite('/NARI/linepoints.jpg', img)
	if key  == ord("p"):
		cv.imwrite('calibrate-1.jpg', img)
	if key == ord("l"):
                cv.imwrite('RetroReflective.jpg', img)
	if key == ord("h"):
		cv.imwrite('nahelem.jpg', img)		
		break
