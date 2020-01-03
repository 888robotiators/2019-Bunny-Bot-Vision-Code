import cv2
import numpy as np
import random
img = cv2.imread("RRDistance.jpg")
npzFile = np.load('cam05.npz')
dist = npzFile['dist']
mtx = npzFile['mtx']
newcameramtx = npzFile['newcameramtx']
roi = npzFile['roi']

img = cv2.undistort(img, mtx, dist, None, newcameramtx)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
frame = img.copy()
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
lowerRed = np.array([0,0, 200])
upperRed = np.array([100,220, 255])
redMask = cv2.inRange(r, lowerRed, upperRed)   
cv2.circle(img, (27, 172), 3, (100, 255, 255), -1)  #yellow first point
cv2.circle(redMask, (27, 172), 3, (100, 255, 255), -1)  #yellow first point
cv2.circle(img, (111, 310), 3, (0, 0, 255), -1)  #red last point
cv2.circle(img, (315, 380), 3, (0, 255, 0), -1)  #green third point
cv2.circle(img, (320, 178), 3, (255, 0, 0), -1)  #blue second point
contours, heirarchy = cv2.findContours(redMask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        contours.remove(c)

if len(contours) > 0:
        c2 = max(contours, key=cv2.contourArea)


bigBox = np.concatenate((c,c2), axis=0)

rect = cv2.minAreaRect(bigBox)
box = cv2.cv.BoxPoints(rect)
box = np.int0(box)
cv2.drawContours(img,[box],0,(0,0,255),2)
focalLength = (rect[1][0] * 20.5) / 8
print focalLength
#print "c" 
#print c
#print "c2"
#print c2
cv2.imshow("image", img)
#cv2.imshow("HSV", hsv)
cv2.imshow("RedMask", redMask)
while True:
    key = cv2.waitKey(25)
    if key == 27:
        break


cv.destroyAllWindows()

