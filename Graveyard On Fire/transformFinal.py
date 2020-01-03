import cv2
import numpy as np
import math

# Notice:
# Everything that is commented with the pink """ comment has not
# been transferred for the new points
from pyimagesearch.transform import four_point_transform

video = cv2.VideoCapture(0)
img = cv2.imread("lineCalibrate3.jpg")

npzFile = np.load('cam02.npz')
dist = npzFile['dist']
mtx = npzFile['mtx']
newcameramtx = npzFile['newcameramtx']
roi = npzFile['roi']

img = cv2.undistort(img, mtx, dist, None, newcameramtx)
#pts = np.array([(104, 193), (383, 197), (626, 340), (8, 388)], dtype = "float32")
pts = np.array([(192, 146), (466, 145), (626, 240), (41, 242)], dtype = "float32")
realWidth = 46.875
realHeight = 55.4375
#print "realWidth"
#print realWidth
print "realHeight"
print realHeight
scalar = 15
scalarWidth = int(realWidth * scalar)
scalarHeight = int(realHeight * scalar)
#print width
#print height
dst = np.array([
    [0, 0],
    [scalarWidth - 1, 0],
    [scalarWidth - 1, scalarHeight - 1],
    [0, scalarHeight - 1]], dtype = "float32")
#print pts
M = cv2.getPerspectiveTransform(pts, dst)
warped = cv2.warpPerspective(img, M, (scalarWidth, scalarHeight))

"""
cv2.circle(img, (104, 193), 3, (100, 255, 255), -1)  #yellow first point
cv2.circle(img, (8, 388), 3, (0, 0, 255), -1)  #red last point
cv2.circle(img, (626, 340), 3, (0, 255, 0), -1)  #green third point
cv2.circle(img, (383, 197), 3, (255, 0, 0), -1)  #blue second point
"""
cv2.circle(img, (192, 146), 3, (100, 255, 255), -1)  #yellow first point
cv2.circle(img, (41, 242), 3, (0, 0, 255), -1)  #red last point
cv2.circle(img, (626, 240), 3, (0, 255, 0), -1)  #green third point
cv2.circle(img, (466, 145), 3, (255, 0, 0), -1)  #blue second point

#perform a gaussian blur on image
#parameters: source, (sigmaX, sigmaY), border type (0 = defualt)
blurred_frame = cv2.GaussianBlur(warped, (5,5), 0)

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
mask2 = mask.copy()
contours, heirarchy = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#contourMat = np.array(contours, dtype = "float32")
#contours.astype(np.float32)

#detect canny edges
#paramters: source, threshold1, threshold2
edges = cv2.Canny(mask, 75, 150)
edges2 = edges.copy()
contours2 = cv2.findContours(edges2, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
cont = np.mat(contours[0], dtype = "float32")
#cv2.drawContours(warped, contours, -1, (0, 165, 255))
#only proceed if at least one contour was found
if len(contours) > 0:
    # find the largest contour in the mask
    c = max(contours, key=cv2.contourArea)

    # finding the extreme values
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    cv2.drawContours(warped, [c], -1, (255,0,0), 3)

cv2.circle(warped, extLeft, 8, (100, 255, 255), -1)  #yellow first point
cv2.circle(warped, extTop, 8, (255, 0, 0), -1)  #blue second point
cv2.circle(warped, extRight, 8, (0, 255, 0), -1)  #green third point
cv2.circle(warped, extBot, 8, (0, 0, 255), -1)  #red last point

"""
print "extLeft"
print extLeft
print "extRight"
print extRight
print "extTop"
print extTop
print "extBot"
print extBot
"""

#warped = four_point_transform(img, pts)
cv2.imshow("image", img)
cv2.imshow("warped", warped)
#cv2.imshow("hsv mask", mask)
#cv2.imshow("contours", mask2)
#cv2.imshow("edges", edges)
#print "contours"
#print type(contours)
#print len(contours)
#print contours

"""
PARTICLE ANALYSIS
"""
diameter = math.sqrt(((extBot[0] - extTop[0]) **2) + ((extBot[1] - extTop[1]) ** 2))
midpoint = (((extBot[0] + extTop[0])/2), ((extBot[1] + extTop[1]) / 2))
trueDiameter = math.sqrt((realHeight ** 2) + 1)
trueCircle = (math.pi * ((trueDiameter / 2) ** 2))
trueRectangle = realHeight
trueRatio = trueRectangle / trueCircle
print "true Diameter"
print trueDiameter
print "trueCircle"
print trueCircle
print "true Rectangle"
print trueRectangle
print "true Ratio"
print trueRatio
print "midpoint"
print midpoint
print "diameter"
print diameter
circleArea = (math.pi * ((diameter / 2) ** 2))
print "circleArea"
print circleArea
retval = cv2.moments(mask2, binaryImage = True)
hu = cv2.HuMoments(retval)
#print retval
#print "Hu 0: MOI?"
#print type(hu[0][0])
#print hu[0][0] # I think this is moment of inertia
#print "contours"
#print type(contours)
#print contours[0]	
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, maxLineGap=50)
avgLine = 0

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        #cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #cv2.circle(new_frame, (x1, y1), 5, (0, 255, 255), -1)  #yellow first point
        #cv2.circle(new_frame, (x2, y2), 5, (0, 0, 255), -1)  #red second point
        radian_angle = math.atan2((x2-x1),(y1-y2))
        degree_angle = math.degrees(radian_angle)

        if degree_angle > 90:
            degree_angle = degree_angle - 180
	    
        avgLine = avgLine + degree_angle
        print degree_angle
    avgLine = avgLine/len(lines)
    math.radians(avgLine)

contour = contours[0][0][0]
#print type(contour)
#print "contour"
contour.astype(np.float32)

area = cv2.contourArea(cont)
print "area"
print area
#print "area"
#print type(area)
#print len(area)
#print area
ratio = float(area/circleArea)
print "ratio"
print ratio



errorInv = 1 - abs(trueRatio - ratio)
print "percent error"
print errorInv


#Convert from pixels to inch
#Decide if the rectangle is correct, based on parameters returned above
#Then find the location and heading of the point



#this is the top left of the rectangle, in real inches
#not yet recalibrated 
#realTopLeft = (-9.25, 89.75) 
realTopLeft = (-23.25, 93.75)


floatScale = float(scalar)
pixelToInch = float(1 / floatScale)
# Formula for converting from pixel to inch:
pixel = (((extBot[0] + extTop[0])/2), ((extBot[1] + extTop[1]) / 2))
print "pixel"
print pixel
print "scalar"
print scalar
print "pixelToInch"
print pixelToInch
inch = (((pixel[0] * pixelToInch) + realTopLeft[0]), (realTopLeft[1] - (pixel[1] * pixelToInch)))
heading = math.atan(inch[0] / inch[1])
math.radians(heading)
print "inch"
print inch
print "heading"
print heading
print "avgLine"
print avgLine


while True:
    key = cv2.waitKey(25)
    if key == 27:
        break


cv.destroyAllWindows()
