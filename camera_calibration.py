
import numpy as np
import cv2
import os
import config

# checkerboard dim
corner_x=config.checkerboard_dim[0]
corner_y=config.checkerboard_dim[1]
block_size = config.checkerboard_dim_block_size

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)


# Mesh grid 
coord = np.zeros((corner_y*corner_x,3), np.float32)
coord[:,:2] = np.mgrid[0:corner_x,0:corner_y].T.reshape(-1,2)
coord*=block_size
Real_word = [] # 3d point in real world space
image_plane = [] # 2d points in image plane.

source_path = config.calibration_img_dir

print(source_path)
images = [os.path.join(source_path,f) for f in os.listdir(source_path)]

for img in images: 
    img_np = cv2.imread(img) # 
    cv2.imshow('img_np', img_np)
    cv2.waitKey(500)
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGRA2GRAY)
    
    # find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (corner_x,corner_y), None)
  
    if ret:
        Real_word.append(coord) 
        corners2 = cv2.cornerSubPix(gray,corners,(20,5),(-1,-5),criteria)
        image_plane.append(corners2)
        img_np = cv2.drawChessboardCorners(img_np, (corner_x,corner_y), corners2, ret)
        cv2.imshow('chessboard', img_np)
        cv2.waitKey(0)
cv2.destroyAllWindows()

#calibration
ret, cmtx, dist_v, Rvecs, Tvecs= cv2.calibrateCamera(Real_word, image_plane, gray.shape[::-1], None, None)

calib_data_path = config.calibration_parameter_dir
if not os.path.exists(calib_data_path):
    os.makedirs(calib_data_path)
    
np.savez(
    f"{calib_data_path}/MultiMatrix",
    camMatrix=cmtx,
    distCoef=dist_v,
    rVector=Rvecs,
    tVector=Tvecs,
)

print("-------------------------------------------")

print(cmtx)
print(dist_v)
print(Rvecs, Tvecs)
print("-------------------------------------------")

