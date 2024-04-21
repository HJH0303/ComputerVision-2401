import cv2, os
import numpy as np
from utils import extract_SIFT, match_descriptors, vis_match

dataset_dir = 'dataset/CMU0'
image_files = sorted(os.listdir(dataset_dir))

#image panorama for two image example
image_dir_left = os.path.join(dataset_dir, image_files[0])
image_dir_right = os.path.join(dataset_dir, image_files[1])

#load images
image_left = cv2.imread(image_dir_left, cv2.IMREAD_COLOR)
image_right = cv2.imread(image_dir_right, cv2.IMREAD_COLOR)
height, width, channels = image_left.shape

#get SIFT keypoint and descriptors 
k_l, d_l, vis_l = extract_SIFT(image_left)
k_r, d_r, vis_r = extract_SIFT(image_right)

# Brute Force matcher
matches = []
ratio_thresh = 0.75
for i, decs_l in enumerate(d_l):
    distances = np.array([np.linalg.norm(decs_l-decs_r, ord=2) for decs_r in d_r])
    idx_sorted = np.argsort(distances)
    if distances[idx_sorted[0]] < ratio_thresh * distances[idx_sorted[1]]:
        matches.append((i, idx_sorted[0]))

mk_l_r = [k_l[0][m[0]] for m in matches]
mk_l_c = [k_l[1][m[0]] for m in matches]
mk_r_r = [k_r[0][m[1]] for m in matches]
mk_r_c = [k_r[1][m[1]] for m in matches]

mk_l, mk_r =[mk_l_c, mk_l_r], [mk_r_c, mk_r_r]

# RANSAC Algorithm
# 1. 4개쌍 match 뽑기
def compute_homography(src_pts, des_pts):
    A= []
    for i in range(0,len(src_pts)):
        x, y = src_pts[i][0],src_pts[i][1]
        u, v = des_pts[i][0],des_pts[i][1]

        A.append([x,y,1,0,0,0, -x*u, -y*u, -u])
        A.append([0,0,0,x,y,1, -x*v, -y*v, -v])
    A =np.array(A)    
    _, _, Vt = np.linalg.svd(A)
    L = Vt[-1,:]/Vt[-1,-1]
    H = L.reshape(3, 3)
    return H
def ransac_homography(mk_l, mk_r, width=width,iterations=1000, threshold=5):
    src_pts = np.transpose(np.array(mk_l))
    des_pts = np.transpose(np.array(mk_r))
    max_inliers = []
    final_H = None
    for i in range(iterations):
        indices = np.random.choice(np.arange(len(src_pts)),4,replace=False)
        src_sample=src_pts[indices]
        des_sample=des_pts[indices]
        H = compute_homography(src_sample,des_sample)
        inliers = []
        for j in range(len(src_pts)):
            point_src =np.append(src_pts[j],1)
            point_dst_estimated = np.dot(H,point_src)
            point_dst_estimated/= point_dst_estimated[-1]
            
            H_inv = np.linalg.inv(H)
            point_dst = np.append(des_pts[j], 1)
            point_src_estimated = np.dot(H_inv, point_dst)
            point_src_estimated /= point_src_estimated[-1]

            if np.linalg.norm(point_dst_estimated-point_dst) <threshold and  np.linalg.norm(point_src_estimated - point_src) <threshold:
                inliers.append(j)
        if len(inliers) > len(max_inliers):
            max_inliers =inliers
            final_H = H
    return final_H , max_inliers
H, inliers = ransac_homography(mk_l, mk_r, width=width) #since stitching right to left is easier.
print(f'number of inliner samples: {len(inliers)}')
print(H)
