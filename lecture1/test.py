import numpy as np
import cv2
def bgr_to_hsv(bgr_image):
    bgr_image = bgr_image / 255.0  # Normalize BGR values to [0, 1]
    b, g, r = bgr_image[..., 0], bgr_image[..., 1], bgr_image[..., 2]

    c_max = np.max(bgr_image, axis=-1)
    c_min = np.min(bgr_image, axis=-1)
    delta = c_max - c_min

    # Hue calculation
    h = np.zeros_like(c_max)
    mask = delta > 0
    idx = (r == c_max) & mask
    h[idx] = (60 * ((g[idx] - b[idx]) / delta[idx]) + 360) % 360
    idx = (g == c_max) & mask
    h[idx] = (60 * ((b[idx] - r[idx]) / delta[idx]) + 120) % 360
    idx = (b == c_max) & mask
    h[idx] = (60 * ((r[idx] - g[idx]) / delta[idx]) + 240) % 360

    # Saturation calculation
    s = np.zeros_like(c_max)
    s[c_max != 0] = delta[c_max != 0] / c_max[c_max != 0]

    # Value calculation
    v = c_max

    hsv_image = np.stack([h, s, v], axis=-1)
    return hsv_image

# Test the function with a dummy BGR image
lenna_image = cv2.imread('Lenna.png')
hsv_image = bgr_to_hsv(lenna_image)

h_channel, s_channel, v_channel = cv2.split(hsv_image)
h_channel_scaled = np.uint8(h_channel * 255 / 360)

# S와 V 채널을 0에서 255로 스케일링합니다.
s_channel_scaled = np.uint8(s_channel * 255)
v_channel_scaled = np.uint8(v_channel * 255)
cv2.imwrite('H_Channel.png', h_channel_scaled)
cv2.imwrite('S_Channel.png', s_channel_scaled)
cv2.imwrite('V_Channel.png', v_channel_scaled)
# # print(R)