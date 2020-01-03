import cv2
import numpy as np
import math

from pyimagesearch.transform import four_point_transform

npzFile = np.load('cam01.npz')
dist = npzFile['dist']
mtx = npzFile['mtx']
newcameramtx = npzFile['newcameramtx']
roi = npzFile['roi']
rotM = npzFile['rotM']
tvec = npzFile['tvec']
cameraPosition = npzFile['cameraPosition']
pxlp = npzFile['pxlp']
realp = npzFile['realp']

video = cv2.VideoCapture(0)
img = cv2.imread("transformTest.jpg")

while True:
#prepare the frame
    #start capturing images from the video
    grabbed, frame = video.read()
    #frame = cv2.undistort(distFrame, mtx, dist, None, newcameramtx)

    if not grabbed:
        print "error"
    pts = np.array([(280, 53), (400, 52), (532, 172), (132, 164)], dtype = "float32")
    width = int(math.sqrt(((172-164) ** 2) + ((532-132) ** 2)))
    height = int(math.sqrt(((172-53) ** 2) + ((532-400) ** 2)))
    realWidth = 8.5 + 5.5 #inches
    realHeight = 83.5 - 24.5 #inches
    scalar = 15
    scalarWidth = int(realWidth * scalar)
    scalarHeight = int(realHeight * scalar)
    dst = np.array([
        [0, 0],
        [scalarWidth - 1, 0],
        [scalarWidth - 1, scalarHeight - 1],
        [0, scalarHeight - 1]], dtype = "float32")
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(frame, M, (scalarWidth, scalarHeight))
    cv2.circle(frame, (280, 53), 5, (100, 255, 255), -1)  #yellow first point
    cv2.circle(frame, (400, 53), 5, (255, 0, 0), -1)  #blue second point
    cv2.circle(frame, (532, 172), 5, (0, 255, 0), -1)  #green third point
    cv2.circle(frame, (132, 164), 5, (0, 0, 255), -1)  #red last point
    cv2.circle(frame, (340, 120), 10, (255, 0, 255), -1) #middle
    cv2.line(frame, (132, 164), (400, 53), (0, 0, 0), 2)
    cv2.line(frame, (280, 53), (530, 172), (0, 0, 0), 2)
    cv2.line(frame, (280, 53), (400, 53), (0, 165, 255), 2)
    cv2.line(frame, (132, 164), (530, 172), (0, 165, 255), 2)
    cv2.line(frame, (280, 53), (132, 164), (0, 165, 255), 2)
    cv2.line(frame, (400, 53), (530, 172), (0, 165, 255), 2)

    
    
    #print pts
    

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
    gray2 = gray.copy()
    contours, heirarchy = cv2.findContours(mask2, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        # find the largest contour in the mask
        c = max(contours, key=cv2.contourArea)

        # finding the extreme values
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])
        cv2.drawContours(warped, [c], -1, (255,0,0), 3)
    #detect canny edges
    #paramters: source, threshold1, threshold2
    edges = cv2.Canny(mask, 75, 150)
    #warped = four_point_transform(img, pts)
    cv2.imshow("image", frame)
    cv2.imshow("warped", warped)
    cv2.imshow("mask", mask)
    cv2.imshow("contours", mask2)
    cv2.imshow("gray", gray)
    cv2.imshow("edges", edges)

    retval = cv2.moments(mask2, binaryImage = True)
    hu = cv2.HuMoments(retval)
    #print retval
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
	
	avgLine = math.radians(avgLine/len(lines))
    #print hu[0][0] # I think this is moment of inertia
    diameter = math.sqrt(((extBot[0] - extTop[0]) **2) + ((extBot[1] - extTop[1]) ** 2))
    realTopLeft = (-5.5, 83.5)
    floatScale = float(scalar)
    pixelToInch = float(1 / floatScale)
    # Formula for converting from pixel to inch:
    pixel = (((extBot[0] + extTop[0])/2), ((extBot[1] + extTop[1]) / 2))
    #print "pixel"
    #print pixel
    #print "scalar"
    #print scalar
    #print "pixelToInch"
    #print pixelToInch
    inch = (((pixel[0] * pixelToInch) + realTopLeft[0]), (realTopLeft[1] - (pixel[1] * pixelToInch)))
    heading = math.atan(inch[0] / inch[1])
    math.radians(heading)
    #print "inch"
    print inch
    #print "heading"
    print heading
    print avgLine
    key = cv2.waitKey(25)
    if key == 27:
        break
video.release()
cv2.destroyAllWindows()
