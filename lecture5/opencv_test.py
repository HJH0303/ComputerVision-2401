import cv2, os
import numpy as np


dataset_dir = 'dataset/CMU0'
image_files = sorted(os.listdir(dataset_dir))

#image panorama for two image example
image_dir_left = os.path.join(dataset_dir, image_files[0])
image_dir_right = os.path.join(dataset_dir, image_files[1])

#load images
image1 = cv2.imread(image_dir_left, cv2.IMREAD_COLOR)
image2 = cv2.imread(image_dir_right, cv2.IMREAD_COLOR)
# SIFT 검출기 초기화
sift = cv2.SIFT_create()

# 각 이미지에서 SIFT 특징점과 기술자 검출
keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

# BFMatcher 객체 생성
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# 특징점 기술자를 이용한 매칭 수행
matches = bf.match(descriptors1, descriptors2)

# 매칭 결과에 따라 거리에 기반한 필터링을 수행하여 좋은 매칭만 선택
matches = sorted(matches, key = lambda x:x.distance)

# 매칭 결과를 이미지에 그리기
result_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:50], None, flags=2)

# 결과 이미지 표시
cv2.imwrite('cv_matcher.png', result_image)
