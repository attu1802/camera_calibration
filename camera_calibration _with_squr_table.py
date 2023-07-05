
import numpy as np
import cv2
import os
import config

# checkerboard dim
corner_x=config.table_dim[0]
corner_y=config.table_dim[1]


# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)


# Mesh grid 
coord = np.zeros((corner_y*corner_x,3), np.float32)
coord[:,:2] = np.mgrid[0:corner_x,0:corner_y].T.reshape(-1,2)

# print(coord)
# store coord.
Real_word = [] # 3d point in real world space
image_plane = [] # 2d points in image plane.

source_path = config.table_calibration_img_dir

images = [os.path.join(source_path,f) for f in os.listdir(source_path)]

def get_corner(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(gray, (7,7), 1) 
    (T, binary_img) = cv2.threshold(img_blur, 220, 255,cv2.THRESH_BINARY)
    corners = cv2.goodFeaturesToTrack(binary_img, 10, 0.01, 10)
    return corners

for img in images: 
    img_np = cv2.imread(img) # 
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGRA2GRAY)
    # find the table  corners
    corners = get_corner(img_np)

    Real_word.append(coord) #Certainly, every loop objp is the same in 3D
    corners2 = cv2.cornerSubPix(gray,corners,(20,5),(-1,-5),criteria)

    image_plane.append(corners2)
    for corner in corners2:
        x,y = corner.ravel()
        print(x,y)
        cv2.circle(img_np, (int(x),int(y)), 2, (0,0,255), 3)
    cv2.imshow('table', img_np)
    cv2.waitKey(0)
cv2.destroyAllWindows()

#calibration
ret, mtx, dist, rvecs, tvecs= cv2.calibrateCamera(Real_word, image_plane, gray.shape[::-1], None, None)

calib_data_path = config.table_calibration_parameter_dir
if not os.path.exists(calib_data_path):
    os.makedirs(calib_data_path)
    
np.savez(
    f"{calib_data_path}/MultiMatrix2",
    camMatrix=mtx,
    distCoef=dist,
    rVector=rvecs,
    tVector=tvecs,
)

print("-------------------------------------------")

print(mtx)
print(dist)
print(rvecs, tvecs)

