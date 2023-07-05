import cv2 
import numpy as np
import math
import config
import os
import random

calib_data_path = config.calibration_parameter_dir+"\MultiMatrix.npz"
calib_data = np.load(calib_data_path)

cam_mat = calib_data["camMatrix"]
dist_coef = calib_data["distCoef"]
r_vectors = calib_data["rVector"]
t_vectors = calib_data["tVector"]


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)

def object_distance(img_pixels,w):
    focal_dis = config.camera_focal_len
    depth = config.depth
    sensor_size = config.sensor_size[0]
    # camera_full_pixel = config.camera_pixel[0]
    camera_full_pixel = w
    dis = depth*img_pixels*sensor_size/(focal_dis*camera_full_pixel)
    return dis

def get_corner(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	checker_board = config.test_img_checkerboard
	if checker_board:
		ret, corners = cv2.findChessboardCorners(gray, (config.test_checkerboard_dim[0],config.test_checkerboard_dim[1]), None)

	else:
		img_blur = cv2.bilateralFilter(gray, 30, 50, 50)
		# cv2.imshow("IMage", binary_img)
		# cv2.waitKey(0)
		# img_blur = cv2.GaussianBlur(gray, (3,3), 1) 
		(T, binary_img) = cv2.threshold(img_blur, 120, 255,cv2.THRESH_BINARY_INV)
		cv2.imshow("IMage", binary_img)
		cv2.waitKey(0)
		corners = cv2.goodFeaturesToTrack(binary_img, 4, 0.01, 20)
		corners = cv2.cornerSubPix(binary_img,corners,(20,5),(-1,-5),criteria)

	pixels_dist =[]
	pixel_point =[]

	for j,corner in enumerate(corners):
		if j == 0:
			x,y = corner.ravel()
			pixel_point.append([int(x),int(y)])
		else:
			x, y = corner.ravel()
			pixel_point.append([int(x),int(y)])
			# print([int(x),int(y)])
			dist = math.sqrt((pixel_point[j-1][0]-pixel_point[j][0])**2 +(pixel_point[j-1][1]-pixel_point[j][1])**2)
			pixels_dist.append(dist)
	pixel_point.append(pixel_point[0])
	dist = math.sqrt((pixel_point[j][0]-pixel_point[0][0])**2 +(pixel_point[j][1]-pixel_point[0][1])**2)
	pixels_dist.append(dist)
	return pixel_point, pixels_dist


img_dir = config.test_imge_path
save_path = config.sav_res
if not os.path.exists(save_path):
	os.makedirs(save_path)

if os.path.isdir(img_dir):
	for img_name in os.listdir(img_dir):
		img_path = os.path.join(img_dir,img_name)
		
		img =cv2.imread(img_path)
		
		h, w = img.shape[:2]
		
		# newcameramtx, roi=cv2.getOptimalNewCameraMatrix(cam_mat, dist_coef , (w,h), 1, (w,h))

		# img = cv2.undistort(img, cam_mat, dist_coef, None, newcameramtx)
		a,b = get_corner(img)
		for i in range(1,len(a)):
			rectangle = np.array([a[i-1],a[i]], np.int32)
			R = random.randint(1,255)
			B = random.randint(5,255)
			G = random.randint(10,255)
			cv2.polylines(img, [rectangle], True, (B,G,R), 3) 
			actual_dis = int(object_distance(b[i-1],w))
			cv2.putText(img, text = str(actual_dis),org = (a[i-1][0]-20,a[i-1][1]-5) ,fontFace = cv2.FONT_HERSHEY_DUPLEX,fontScale = 2,color = (B, G, R),thickness = 2)
		img_save_name = str(random.randint(5000,80000))
		cv2.imwrite(os.path.join(save_path,img_save_name+".jpeg"), img)
		

elif os.path.isfile(img_dir):
		img =cv2.imread(img_dir)
		h, w = img.shape[:2]

		# newcameramtx, roi=cv2.getOptimalNewCameraMatrix(cam_mat, dist_coef , (w,h), 1, (w,h))
	
		# img = cv2.undistort(img, cam_mat, dist_coef, None, newcameramtx)
		a,b = get_corner(img)
		for i in range(1,len(a)):
			rectangle = np.array([a[i-1],a[i]], np.int32)
			R = random.randint(1,255)
			B = random.randint(5,255)
			G = random.randint(10,255)
			cv2.polylines(img, [rectangle], True, (B,G,R), 3) 
			actual_dis = int(object_distance(b[i-1],w))
			cv2.putText(img, text = str(actual_dis),org = (a[i-1][0]+20,a[i-1][1]-5) ,fontFace = cv2.FONT_HERSHEY_DUPLEX,fontScale = 2,color = (B, G, R),thickness = 2)
		img_save_name = str(random.randint(5000,80000))
		cv2.imwrite(os.path.join(save_path,img_save_name+".jpeg"), img)

cv2.destroyAllWindows()
		

