import cv2
import numpy as np

# 이미지를 불러옵니다.
image = cv2.imread('eve.jpg')

# SIFT 특징점 검출기를 생성합니다.
sift = cv2.SIFT_create(nfeatures=500, contrastThreshold=0.04)

# 이미지에서 특징점과 기술자를 검출합니다.
keypoints, descriptors = sift.detectAndCompute(image, None)

# 특징점을 시각화합니다. drawKeypoints 함수를 사용하여 특징점을 이미지에 그립니다.
keypoint_image = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# 시각화된 이미지를 보여줍니다.
cv2.imwrite("eve_SIFT_cv.png",keypoint_image)
