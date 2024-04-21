import cv2
import numpy as np

# 이미지 로드
# lenna_image_gray = cv2.imread('gray_lenna.png', cv2.IMREAD_GRAYSCALE).astype(np.float64)

lenna_image_gray = cv2.imread('eve.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float64)
gaussain_kernel = 1/16*np.array([[1.0,2.0,1.0],
                                 [2.0,4.0,2.0],
                                 [1.0,2.0,1.0]])

# 가우시안 필터 적용
lenna_gaussian = cv2.filter2D(lenna_image_gray, -1, gaussain_kernel, borderType=cv2.BORDER_CONSTANT)

# sobel filter 적용 후 normalize
Ix = (1/8)*np.array([[-1.0 ,0.0, 1.0],
              [-2.0 ,0.0, 2.0],
              [-1.0 ,0.0, 1.0]])

Iy = (1/8)*np.array([[1.0 ,2.0, 1.0],
              [0.0 ,0.0, 0.0],
              [-1.0 ,-2.0, -1.0]])

lenna_Ix = cv2.filter2D(lenna_gaussian, -1, Ix, borderType=cv2.BORDER_CONSTANT)
lenna_Iy = cv2.filter2D(lenna_gaussian, -1, Iy, borderType=cv2.BORDER_CONSTANT)

grad_mag = np.sqrt(lenna_Ix*lenna_Ix+lenna_Iy*lenna_Iy)
grad_ori = np.arctan2(lenna_Iy,lenna_Ix)

# Non Maximal Suppression
mag_255 = grad_mag/np.max(grad_mag)*255
ori_0_360 = grad_ori*180./np.pi
ori_0_360[ori_0_360<0]+=180
M = len(lenna_image_gray)
N = len(lenna_image_gray[0])

nms_result = np.zeros((M,N), dtype=np.float64) # resultant image
for i in range(1,M-1):
    for j in range(1,N-1):
        r= 255.
        q= 255.
        if (0.0<=ori_0_360[i][j]<22.5) or (157.5<=ori_0_360[i][j]<180.):
            r= mag_255[i][j-1]
            q= mag_255[i][j+1]

        elif 22.5<=ori_0_360[i][j]<67.5:
            r = mag_255[i-1, j+1]
            q = mag_255[i+1, j-1]
        
        elif 67.5<=ori_0_360[i][j]<112.5:
            r= mag_255[i-1][j]
            q= mag_255[i+1][j]

        elif 112.5<=ori_0_360[i][j]<157.5:
            r= mag_255[i+1][j+1]
            q= mag_255[i-1][j-1]
        if mag_255[i][j]>=q and mag_255[i][j]>=r: 
            nms_result[i][j] = mag_255[i][j]
        else:
            nms_result[i][j]=0

# double threshold
lowThresholdRatio = 0.05
highThresholdRatio = 0.09
weak_default=25
strong_default=255
highThreshold = nms_result.max() * highThresholdRatio
lowThreshold = highThreshold * lowThresholdRatio
weak = np.int32(weak_default)
strong = np.int32(strong_default)
res = np.zeros((M,N), dtype=np.int32)

weak_i, weak_j = np.where(nms_result<lowThreshold)
zeros_i, zeros_j = np.where(nms_result < lowThreshold)
strong_i, strong_j = np.where(nms_result>highThreshold)
res[strong_i,strong_j] = strong
res[weak_i,weak_j] = weak

for i in range(1, M-1):
    for j in range(1, N-1):
        if (res[i,j] == weak):
            if ((nms_result[i][j-1] == strong) or (nms_result[i-1][j-1] == strong) or
                (nms_result[i-1][j] == strong) or (nms_result[i-1][j-1] == strong) or
                (nms_result[i][j+1] == strong) or (nms_result[i+1][j+1] == strong) or
                (nms_result[i+1][j] == strong) or (nms_result[i+1][j-1] == strong) ):
                res[i,j] = strong
            else:
                res[i,j] = 0
cv2.imwrite('eve_double_Threshold.png', res) # sharpen edges..
