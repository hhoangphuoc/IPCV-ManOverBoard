import numpy as np
import cv2 as cv
import glob
import matplotlib.pyplot as plt

frameSize = (640,480)
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

 
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
 

images = glob.glob('./calibration images/*.jpg')
 
for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv.findChessboardCorners(gray, (9, 6), None)

    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)  # Append only if corners are found
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)  # Append refined corners

        # Draw and display the corners
        cv.drawChessboardCorners(img, (9, 6), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(350)

 
cv.destroyAllWindows()
#*********************************************************************************#
# Calibration
ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)

# Intrinsic Parameters
print("Camera Matrix (Intrinsic Parameters):")
print(cameraMatrix)

# Distortion Coefficients
print("Distortion Coefficients:")
print(dist)

# Extrinsic Parameters (for each calibration image)
for i in range(len(rvecs)):
    print(f"Rotation Vector for Image {i+1}:")
    print(rvecs[i])
    print(f"Translation Vector for Image {i+1}:")
    print(tvecs[i])

#*********************************************************************************#
img = cv.imread('picForCalibration.jpg')
h,  w = img.shape[:2]
newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))

# Undistort
dst = cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('caliResult1.png', dst)

# Undistort with Remapping
mapx, mapy = cv.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w,h), 5)
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('caliResult2.png', dst)

#*********************************************************************************#
# Reprojection Error
mean_error = 0
errors = []  # List to hold individual reprojection errors

for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
    errors.append(error)  # Store individual error
    mean_error += error

print("Total reprojection error: {}".format(mean_error/len(objpoints)))


