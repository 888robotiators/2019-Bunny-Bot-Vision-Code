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


extBot = [pxlp[3][0], pxlp[3][1], 1]
extLeft = [pxlp[0][0], pxlp[0][1], 1]

extBot = np.mat(extBot).T.reshape(3,1)
extLeft = np.mat(extLeft).T.reshape(3,1)


realBot = [-4.5, 27.385, 0, 1]
realBot = np.mat(realBot).T.reshape(4,1)
realLeft = [-24.277, 52.37, 0, 1]
realLeft = np.mat(realLeft).T.reshape(4,1)

# 3x4 combination of rvecs and tvecs
extrinsicParameters = np.array([[rotM[0,0], rotM[0,1], rotM[0,2], tvec[0][0]], [rotM[1,0], rotM[1,1], rotM[1,2], tvec[1][0]], [rotM[2,0], rotM[2,1], rotM[2,2], tvec[2][0]]])

# 4x4 homogenous combination of rvecs and tvecs
homogenous = np.array([[rotM[0,0], rotM[0,1], rotM[0,2], tvec[0][0]], [rotM[1,0], rotM[1,1], rotM[1,2], tvec[1][0]], [rotM[2,0], rotM[2,1], rotM[2,2], tvec[2][0]], [0, 0, 0, 1]])
#print homogenous

worldToCamera = np.mat(homogenous)
#print worldToCamera

cameraToWorld2 = np.linalg.inv(worldToCamera)

cameraToWorld = worldToCamera.T
#print cameraToWorld

augRot = np.array([[rotM[0,0], rotM[0,1], rotM[0,2], 0], [rotM[1,0], rotM[1,1], rotM[1,2], 0], [rotM[2,0], rotM[2,1], rotM[2,2], 0], [0, 0, 0, 1]])
augRot = np.mat(augRot)
#print augRot
#print augRot.shape

augT = np.array([[1, 0, 0, tvec[0][0]], [0, 1, 0, tvec[1][0]], [0, 0, 1, tvec[2][0]], [0, 0, 0, 1]])
#augT = np.array([[1, 0, 0, cameraPosition[0][0]], [0, 1, 0, cameraPosition[1][0]], [0, 0, 1, cameraPosition[2][0]], [0, 0, 0, 1]])
augT = np.mat(augT)


combined = augRot * augT
#print combined

pixel = [pxlp[3][0], pxlp[3][1], 0, 1]
pixel = np.mat(pixel).T.reshape(4,1)



left = np.linalg.inv(rotM) * np.linalg.inv(mtx) * extBot
right = np.linalg.inv(rotM) * tvec

print right[2][0]
print type(right)
print left[2][0][0]
print type(left)

s = (14 + right[2][0]) / left[2][0]
print "s"
print type(s)
print s

s = -44.81862757

real4 = np.linalg.inv(rotM) * (s * np.linalg.inv(mtx) * extBot - tvec)


real = cameraToWorld * pixel
real2 = augRot * augT * pixel
real3 = cameraToWorld2 * pixel
#print real
#print real2
#print real3
print real4
#print cameraPosition





