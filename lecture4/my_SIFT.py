import numpy as np
import cv2
from utils import vis_image, get_octave_num, build_gaussian_kernels, generate_gaussian_images, generate_DoG_images
from utils import find_scale_space_extrema, visualize_keypoints_with_orientation
from utils import assign_keypoint_orientation_gaussian, generate_sift_descriptor
from utils import compute_HoG_descriptor, visualize_HoG


lenna_image_bgr = cv2.imread('bgr_lenna.png', cv2.IMREAD_COLOR)
lenna_image_gray = cv2.cvtColor(lenna_image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

lenna_image_gray = cv2.GaussianBlur(lenna_image_gray, (5, 5), sigmaX=1.6, sigmaY=1.6)

num_octaves = get_octave_num(lenna_image_gray)  # number of octave
num_layers = 3  # number of scale for each octave
initial_sigma = 1.6  # initial sigma

# generate simga_values
k = 2**(1/num_layers)
simga_values = np.array([initial_sigma*k**i for i in range(0,num_layers+3)])

# generate gaussian_images

gaussian_images = []
lenna_image_octave = lenna_image_gray
for i in range(num_octaves):
    octave_images = [cv2.GaussianBlur(lenna_image_octave, (0, 0), sigmaX=sigma_i) for sigma_i in simga_values]
    lenna_image_octave=cv2.pyrDown(lenna_image_octave)
    gaussian_images.append(octave_images)

# generate DoG images
DoG_images = []
for octave_images in gaussian_images:
    DoG = [cv2.subtract(octave_images[layer+1],octave_images[layer]) for layer in range(0, len(octave_images)-1)]
    DoG_images.append(DoG)
# find scale extrema
image_border_width=5
contrast_threshold=0.04
prelim_contrast_threshold = np.floor(0.5 * contrast_threshold / num_layers * 255) 
def is_extrema(DoG_images,i,j,octave,layer):
    pixel_value = DoG_images[octave][layer][i, j]
    # print(prelim_contrast_threshold)
    if abs(pixel_value) < prelim_contrast_threshold:
        return False
    
    for x in range(i-1,i+2):
        for y in range(j-1,j+2):
            if x <0 or y <0 or x >= DoG_images[octave][layer].shape[0] or y >= DoG_images[octave][layer].shape[1]: continue
            if layer>0 and DoG_images[octave][layer-1][x,y] >= pixel_value: return False
            if layer<=len(DoG_images[octave])-1 and DoG_images[octave][layer+1][x,y] >= pixel_value: return False
            if (x != i or y != j) and DoG_images[octave][layer][x,y] >= pixel_value: return False
    return True

octaves = []
layers = []
keypoints_h = []
keypoints_w = []
for octave_, octave_images in enumerate(DoG_images):
    for layer_ in range(1,num_layers+1):
        for i in range(image_border_width,octave_images[layer_].shape[0]-image_border_width):
            for j in range(image_border_width,octave_images[layer_].shape[1]-image_border_width):
                if is_extrema(DoG_images,i,j,octave_,layer_):
                    octaves.append(octave_)
                    layers.append(layer_)
                    keypoints_h.append(i)
                    keypoints_w.append(j)

keypoints = [octaves,layers,keypoints_h,keypoints_w]

'''
assign_keypoint_orientation_gaussian
'''
def calculate_gradient(image):
    h_x= 1/8*np.array([[-1.0, 0.0, 1.0],
                       [-2.0, 0.0, 2.0],
                       [-1.0, 0.0, 1.0]])
    h_y= 1/8*np.array([[1.0, 2.0, 1.0],
                       [0.0, 0.0, 0.0],
                       [-1.0, -2.0, -1.0]])
    grad_x = cv2.filter2D(image,-1,h_x, borderType=cv2.BORDER_CONSTANT)
    grad_y = cv2.filter2D(image,-1,h_y, borderType=cv2.BORDER_CONSTANT)
    mag, ori = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=True)
    return mag, ori
def create_orientation_histogram_gaussian(mag_patch, ori_patch, weight, num_bins=36):
    histogram = np.zeros(num_bins)
    bin_width = 360//num_bins
    for mag, ori, w in zip(mag_patch.flatten(), ori_patch.flatten(), weight.flatten()):
        bin_idx = int(np.floor(ori/bin_width)) % num_bins
        histogram[bin_idx]+=mag*w
    return histogram

def gaussian_weighted_histogram(image,keypoint,num_bins=36,scale=1.5):
    _, _, kpt_i, kpt_j = keypoint
    radius = int(3 * scale)
    weight_sigma = 1.5 * scale
    img_patch = image[max(0, kpt_i-radius):kpt_i+radius+1, max(0, kpt_j-radius):kpt_j+radius+1]
    if img_patch.size == 0:
        return np.zeros(num_bins), 0

    mag_patch, ori_patch = calculate_gradient(img_patch)

    y, x = np.indices((img_patch.shape))
    
    distances = np.sqrt((radius-x)**2+(radius-y)**2)
    weight = np.exp(-(distances**2) / (2 * (weight_sigma**2)))
    weight = weight / (weight_sigma * np.sqrt(2 * np.pi))
    histogram = create_orientation_histogram_gaussian(mag_patch, ori_patch, weight, num_bins)
    dominant_orientation = np.argmax(histogram) * (360/num_bins)
    return histogram, dominant_orientation

keypoints_with_ori = []
for keypoint in np.array(keypoints).transpose():
    octave, layer, i ,j = keypoint
    image = gaussian_images[octave][layer]
    _, dominant_orientation = gaussian_weighted_histogram(image,(octave,layer,i,j))
    keypoints_with_ori.append([octave, layer, i, j, dominant_orientation])

# generate_sift_descriptor
def calculate_discriptor(patch, num_bins, width,max_val=0.2, eps=1e-7):
    patch_size = patch.shape[0]
    bin_width = 360 // num_bins
    descriptor = np.zeros((width*width*num_bins),dtype=np.float32)

    subregion_size = patch_size // width
    mag, ori = calculate_gradient(patch)
    for i in range(width):
        for j in range(width):
            start_i = i * subregion_size
            start_j = j * subregion_size
            end_i = start_i + subregion_size
            end_j = start_j + subregion_size

            mag_sub = mag[start_i:end_i,start_j:end_j].flatten()
            ori_sub = ori[start_i:end_i,start_j:end_j].flatten()

            for mag_, ori_ in zip(mag_sub,ori_sub):
                bin_idx = int(np.floor(ori_)//bin_width)%num_bins
                descriptor[(i * width + j) * num_bins + bin_idx] += mag_
    descriptor /= (np.linalg.norm(descriptor) + eps)

    # 값이 매우 큰 특징을 제한
    descriptor = np.minimum(descriptor, max_val)
    descriptor /= (np.linalg.norm(descriptor) + eps)  
    return descriptor
descriptors = []
width = 4
num_bins = 8
for keypoint in keypoints_with_ori:
    octave, layer, i, j, dominant_orientation = keypoint
    img = gaussian_images[octave][layer]
    angle = -dominant_orientation
    rotation_matrix = cv2.getRotationMatrix2D((float(j), float(i)), angle, 1)
    rotated_img = cv2.warpAffine(img,rotation_matrix,(img.shape[1], img.shape[0]))

    # make subregion
    half_patch_size = width * num_bins // 2
    center = half_patch_size
    
    if (i-center<0 or j-center<0 or i+center>=rotated_img.shape[0] or j+center>=rotated_img.shape[1]): continue

    patch = rotated_img[i-center:i+center, j-center:j+center]
    if patch.shape[0] == width*num_bins and patch.shape[1] == width*num_bins:
        descriptor = calculate_discriptor(patch, num_bins, width)
        descriptors.append(descriptor)
descriptor = np.array(descriptors)

lenna_vis = visualize_keypoints_with_orientation(lenna_image_bgr, keypoints_with_ori)
cv2.imwrite('lenna_SIFT2.png', lenna_vis)
