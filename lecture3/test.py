import cv2
import numpy as np

# 이미지 로드
# lenna_image_gray = cv2.imread('gray_lenna.png', cv2.IMREAD_GRAYSCALE).astype(np.float64)
eve_gray = cv2.imread('eve.jpg')

# eve_double_Threshold = cv2.imread('eve_double_Threshold.png', cv2.IMREAD_GRAYSCALE).astype(np.float64)
# cv2.imwrite('eve_final.png', eve_gray+eve_double_Threshold) # sharpen edges..
# eve_gray = cv2.resize(eve_gray, dsize=(720, 720))
eve_canny = cv2.Canny(eve_gray,60,200) 
cv2.imwrite('eve_final2.png', eve_canny) # sharpen edges..
# cv2.imshow('After', eve_canny)
# cv2.waitKey(0) 