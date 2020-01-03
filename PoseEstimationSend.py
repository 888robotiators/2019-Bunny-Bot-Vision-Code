# import the necessary packages
from collections import deque
import numpy as np
import argparse
import imutils
import cv2
import struct
import socket

npzFile = np.load('cam0M.npz')
dist = npzFile['dist']
mtx = npzFile['mtx']
newcameramtx = npzFile['newcameramtx']
roi = npzFile['roi']

#defines udp server address for jetson
HOST = '10.8.88.19'
PORT = 5809

#sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#sock.bind((HOST, PORT))

rioAddress = ('10.8.88.2', 5805)
camera = cv2.VideoCapture(0)



# keep looping
while True:
	# grab the current frame
        (grabbed, frame) = camera.read()
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
        """
        lowerRed = np.array([0,0, 170]) 
        upperRed = np.array([130,100, 255])
        frame = cv2.GaussianBlur(frame, (3,3), 0) # This help remove artifacting. Makes it slightly more accurate.
        redMask = cv2.inRange(frame, lowerRed, upperRed)
        frame2 = frame.copy()

        """
        perform hsv mask on image
        def onmouse(k, x, y, s, p):
            global frame
            if k==1:
                print frame[y,x]
        cv2.namedWindow("frame")
        cv2.setMouseCallback("frame",onmouse)
        """
        contours, heirarchy = cv2.findContours(redMask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        objP = np.array([(-4, 2.5, 0), (-4, -2.5, 0),(-2, 2.5, 0), (-2, -2.5, 0),(2, 2.5, 0) , (2, -2.5, 0), (4, 2.5, 0), (4, -2.5, 0)], dtype = np.float32())
        #print "contours"
        #print len(contours)
        if len(contours) > 0:
                allApprox = None
                for c in contours:
                        epsilon = 0.1*cv2.arcLength(c,True)
                        approx = cv2.approxPolyDP(c,epsilon,True)
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
                        cv2.drawContours(frame2, [sortApprox], 0, (0, 255, 0), 1) # Comment out during actual run
                        sortApprox = sortApprox.astype(np.float32)
                        retval, rvec, tvec = cv2.solvePnP(objP, sortApprox, mtx, None)
                        print rvec
                        print ""
                        print tvec
                        print ""
                        inches = tvec[2]

                        cv2.putText(frame2, "%.2fft" % (inches / 12), (frame2.shape[1]-300, frame2.shape[0]-20), cv2.FONT_HERSHEY_TRIPLEX, 2.0, (0, 255, 0), 3)
        """
        if otherAddress is not None:
                if rvec is not None and tvec is not None:
                        send = struct.pack("!ffffff", float(tvec[0]), float(tvec[1]), float(tvec[2]), float(rvec[0]), float(rvec[1]), float(rvec[2]))
                        sock.sendto(send, rioAddress)
                else:
                        send = struct.pack("!ffffff", -1.0, -1.0, -1.0, -1.0, -1.0, -1.0)
                        sock.sendto(send, rioAddress)
        """
	#cv2.imshow("frame", frame)
        cv2.imshow("frame2", frame2) # Comment out during actual run

        key = cv2.waitKey(1) & 0xFF

        # if the 'q' key is pressed, stop the loop
        if key == ord("q"):
                break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
