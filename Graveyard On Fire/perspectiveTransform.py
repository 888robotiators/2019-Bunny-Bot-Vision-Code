import cv2
import numpy as np

npzFile = np.load('cam01UD.npz')
dist = npzFile['dist']
mtx = npzFile['mtx']
newcameramtx = npzFile['newcameramtx']
roi = npzFile['roi']
rotM = npzFile['rotM']
tvec = npzFile['tvec']
cameraPosition = npzFile['cameraPosition']
pxlp = npzFile['pxlp']
realp = npzFile['realp']

print pxlp
print realp
print type(pxlp)
print type(realp)
print pxlp.dtype
print realp.dtype

src = np.array([[153, 332], [276, 270], [383, 300], [275, 378]], np.float32)
dst = np.array([[-9.14314556, 24.60571861], [-3.1828053, 33.85095215], [3.96123862, 29.24523354], [-1.999102, 20]], np.float32)

extBot = [pxlp[3][0], pxlp[3][1], 1]
extLeft = [pxlp[0][0], pxlp[0][1], 1]

extBot = np.mat(extBot).T.reshape(3,1)
extLeft = np.mat(extLeft).T.reshape(3,1)


realBot = [-4.5, 27.385, 0, 1]
realBot = np.mat(realBot).T.reshape(4,1)
realLeft = [-24.277, 52.37, 0, 1]
realLeft = np.mat(realLeft).T.reshape(4,1)


image2world = cv2.getPerspectiveTransform(src, dst)
print image2world

world2image = np.linalg.inv(image2world)
print world2image

'''
#worked with z=1????
real = [[-9.14], [24.61], [0]]
real = np.mat(real)
print real

pixel = world2image * real
print pixel
'''

pixel = [[153], [332], [1]]
pixel = np.mat(pixel)

testPixel = [[255], [162], [1]]
testPixel = np.mat(testPixel)

#testPixel = [[215], [355], [1]]
#testPixel = np.mat(testPixel)


real = image2world * testPixel
print real






