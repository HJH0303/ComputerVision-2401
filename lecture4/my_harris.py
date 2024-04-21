import cv2
import numpy as np
import math
import random 
from utils import harris_response, detect_corners, vis_image, get_normalized_image_gray

lenna_image_bgr = cv2.imread('eve.jpg', cv2.IMREAD_COLOR)
lenna_image_gray = cv2.cvtColor(lenna_image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
h_x = 1/8*np.array([[-1.0, 0.0 ,1.0],
                     [-2.0, 0.0 ,2.0],
                     [-1.0, 0.0 ,1.0]])

h_y = 1/8*np.array([ [1.0,   2.0 , 1.0],
                     [0.0,   0.0 , 0.0],
                     [-1.0, -2.0 ,-1.0]])

gradinet_x = cv2.filter2D(lenna_image_gray, -1, h_x, borderType=cv2.BORDER_CONSTANT)
gradinet_y = cv2.filter2D(lenna_image_gray, -1, h_y, borderType=cv2.BORDER_CONSTANT)

Ixx = gradinet_x**2
Iyy = gradinet_y**2
Ixy = gradinet_x*gradinet_y

h_g = (1/16)*np.array([[1.0, 2.0, 1.0],
                        [2.0, 4.0, 2.0],
                        [1.0, 2.0, 1.0]])
Ixx = cv2.filter2D(Ixx, -1, h_g, borderType=cv2.BORDER_CONSTANT)
Iyy = cv2.filter2D(Iyy, -1, h_g, borderType=cv2.BORDER_CONSTANT)
Ixy = cv2.filter2D(Ixy, -1, h_g, borderType=cv2.BORDER_CONSTANT)

det = Ixx*Iyy-Ixy**2
trace = Ixx+Iyy
theta = det - 0.04*trace**2
corner_response = theta>1e-5

lenna_keypoint = np.nonzero(corner_response)
lenna_vis = vis_image(lenna_image_bgr, lenna_keypoint)
cv2.imwrite('eve_harris.png', lenna_vis)
