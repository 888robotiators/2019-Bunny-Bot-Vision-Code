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
HOST = '0.0.0.0'
#HOST = '10.8.88.19'
PORT = 5809
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((HOST, PORT))

degree_angle = None
increment    = 0
lowerWhite   = np.array([0, 0, 220])
#lowerWhite   = np.array([0, 0, 180])
message      = None
npzFile      = np.load('cam01.npz')
nullMessage  = struct.pack('!ffff', -1.0, -1.0, -1.0, -1.0)
#pts          = np.array([(172, 218),
#                         (404, 212),
#                         (525, 325),
#                         (41, 334)],
#                         dtype = "float32")
pts          = np.array([(363, 176),
                         (574, 198),
                         (515, 430),
                         (111, 310)],
                         dtype = "float32")
realHeight   = 33
#realHeight   = 28.5
realTopLeft  = (-9.25, 42)
#realTopLeft  = (-16.5, 57)
realWidth    = 18.5
#realWidth   = 27.5
roi          = npzFile['roi']
rioAddress   = ('10.8.88.2', 5805)
scalar       = 16
scalarWidth  = int(realWidth  * scalar)
scalarHeight = int(realHeight * scalar)
upperWhite   = np.array([60, 200, 255])
#upperWhite   = np.array([160, 235, 255])
video        = cv2.VideoCapture(0)
#video        = cv2.VideoCapture(1)


floatScale   = float(scalar)
pixelToInch  = float(1 / floatScale)
dst          = np.array([[0,               0],
                         [scalarWidth - 1, 0],
                         [scalarWidth - 1, scalarHeight - 1],
                         [0,               scalarHeight - 1]],
                         dtype = "float32")
M            = cv2.getPerspectiveTransform(pts, dst)
dist         = npzFile['dist']
mtx          = npzFile['mtx']
newcameramtx = npzFile['newcameramtx']
otherAddress = rioAddress

while True:

    # Capture one image frame from the video source
    grabbed, frame = video.read()

    if not grabbed:
        continue

    # Undistort the frame to correct pixel location erros
    frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)

    # Transform the undistored frame to create a "Top Down" view
    warped = cv2.warpPerspective(frame, M, (scalarWidth, scalarHeight))

    #cv2.circle(frame, (363, 176), 5, (100, 255, 255), -1)  # yellow first point
    #cv2.circle(frame, (111, 310), 5, (0, 0, 255), -1)      # red last point
    #cv2.circle(frame, (515, 430), 5, (0, 255, 0), -1)      # green third point
    #cv2.circle(frame, (574, 196), 5, (255, 0, 0), -1)      # blue second point

    # Convert the "Top Down" view to HSV colors for better masking results
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)

    # Create a "White" range mask to identify possible targets
    mask = cv2.inRange(hsv, lowerWhite, upperWhite)

    # Use findCountours to locate possible targets
    contours, heirarchy = cv2.findContours(mask,
                                           cv2.RETR_LIST,
                                           cv2.CHAIN_APPROX_NONE)

    # FOR DEBUG ONLY
    #cv2.imshow("image", frame)
    #cv2.imshow("mask", mask)
    #cv2.imshow("hsv", hsv)

    # If at least one possible target exists...
    if len(contours) > 0:

        # Identify the largest contour in the mask as the desired target
        targetCountour = max(contours, key=cv2.contourArea)

        # Place a bounding box around the desired target
        # NOTE: minAreaRect is utilized here to allow for rotated bounding
        #       box of smaller area
        rect = cv2.minAreaRect(targetCountour)

        # Convert the rectangle into 4 box points
        box = cv2.cv.BoxPoints(rect)

        # Make sure the 4 box points are numpy integers
        box = np.int0(box)

        # FOR DEBUG ONLY
        #cv2.drawContours(warped,[box],0,(0,0,255),2)
        #print('Box: {0}'.format(box))

        # Determine the distance of one side of the box
        dist0 = math.sqrt((box[0][0] - box[1][0]) ** 2 +
                          (box[0][1] - box[1][1]) ** 2)

        # Determine the distance of one side of the box that
        # is perpendicular to the the side used for dist0
        dist1 = math.sqrt((box[1][0] - box[2][0]) ** 2 +
                          (box[1][1] - box[2][1]) ** 2)

        # Select the short distance as it represents the end of the target
        if (dist0 < dist1):

            # Find the center points on the short ends of the target
            shortEnd1MidPoint = ((box[0][0] + box[1][0]) / 2,
                                 (box[0][1] + box[1][1]) / 2)

            shortEnd2MidPoint = ((box[2][0] + box[3][0]) / 2,
                                 (box[2][1] + box[3][1]) / 2)

        else:

            shortEnd1MidPoint = ((box[0][0] + box[3][0]) / 2,
                                 (box[0][1] + box[3][1]) / 2)

            shortEnd2MidPoint = ((box[1][0] + box[2][0]) / 2,
                                 (box[1][1] + box[2][1]) / 2)

        # The short end point with the higher Y identifies the
        # end point closer to the robot (i.e.: the target way point)
        if (shortEnd1MidPoint[1] > shortEnd2MidPoint[1]):

            target = shortEnd1MidPoint

        else:

            target = shortEnd2MidPoint

        angle = math.atan2((shortEnd2MidPoint[0] - shortEnd1MidPoint[0]),
                           (shortEnd1MidPoint[1] - shortEnd2MidPoint[1]))

        # If the angle exceeds 90 degrees... convert it to the previous
        # circle revolution by subtracting 180 degrees. This works because
        # the target angle exist in the area of about -90 degrees to +90
        # degress
        if angle > np.pi:

            angle -= (np.pi * 2.0)

        # Convert the target alignment angle from radians to degrees
        angle = angle * 180.0 / math.pi

        # FOR DEBUG ONLY
        #print('angle:  {0}'.format(angle))
        #print('target: {0}'.format(target))
        #cv2.circle(warped, shortEnd1MidPoint, 7, (0, 166, 255), -1)  #yellow first point
        #cv2.circle(warped, shortEnd2MidPoint, 7, (0, 166, 255), -1)  #yellow first point
        #cv2.circle(warped, tuple(box[0]), 7, (0, 255, 0), -1)
        #cv2.circle(warped, tuple(box[1]), 7, (255, 0, 0), -1)
        #cv2.circle(warped, tuple(box[2]), 7, (0, 0, 255), -1)
        #cv2.circle(warped, tuple(box[3]), 7, (255, 165, 0), -1)
        #cv2.circle(warped, target, 7, (0, 0, 0), -1)  #black point

        # Convert the target point from pixels to inches
        target = (((target[0] * pixelToInch) + realTopLeft[0]),
                  (realTopLeft[1] - (target[1] * pixelToInch)))

        # Format the data for sending to the RoboRio
        message = struct.pack('!ffff',
                              float(increment),
                              float(target[0]),
                              float(target[1]),
                              float(angle))

        # Send message to RoboRio
        try:
            sock.sendto(message, otherAddress)
        except Exception:
            pass

        # Increment frame count counter
        increment += 1

    # If at least one possible target exists...
    else:

        # Send null message to RoboRio to indicate no current target
        try:
            sock.sendto(nullMessage, otherAddress)
        except Exception:
            pass

    # FOR DEBUG ONLY
    #cv2.imshow("warped", warped)

    key = cv2.waitKey(25)
    if key == 27:
        break

video.release()
cv2.destroyAllWindows()


#print "Looking"
#while otherAddress is None:
#    try:
#
#        cycle, otherAddress = sock.recvfrom(65507)
#        print cycle
#        print otherAddress
#
#    except Exception:
#        cycle = None
#        rioAddress = None

#trueInch = (((truePixel[0] * pixelToInch) + realTopLeft[0]),
#             (realTopLeft[1] - (truePixel[1] * pixelToInch)))

#print float(increment), float(target[0]), float(target[1]), float(angle)
