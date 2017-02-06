import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

from processing import calibrate_camera, undistort_img, colorspace_threshold, gradient_threshold, \
                       transform_perspective, histogram, sliding_window


ret, M, dist, rvecs, tvecs = calibrate_camera('\\camera_cal\\', 9, 6)

test_im = os.getcwd() + '\\test_images\\test5.jpg'
im = cv2.imread(test_im)
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
undistorted = undistort_img(im, M, dist)

color_thresh = colorspace_threshold(undistorted, cv2.COLOR_RGB2HSV, 2, (215, 255))
x_grad = gradient_threshold(undistorted, cv2.COLOR_RGB2HSV, 2, 'x', (10, 30), 3)

thresh_img = np.zeros_like(x_grad)
thresh_img[(x_grad == 1) | (color_thresh == 1)] = 1

# plt.imshow(thresh_img, cmap='gray')

ymax, xmax = thresh_img.shape[:2]
x_shift = 80
src = np.float32([
    [0, 675],
    [xmax//2 - x_shift, 450],
    [xmax//2 + x_shift, 450],
    [xmax, 675]
  ])

shift = 100
dst = np.float32([
    [shift, ymax],
    [shift, shift],
    [xmax-shift, shift],
    [xmax-shift, ymax]
  ])

top_down = transform_perspective(thresh_img, (xmax, ymax), src, dst)
left_fit, right_fit = sliding_window(top_down, 9)

ploty = np.arange(ymax, dtype=np.float32)
left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

plt.imshow(top_down, cmap='gray')
plt.plot(left_fitx, ploty, color='blue')
plt.plot(right_fitx, ploty, color='blue')

warp_zero = np.zeros_like(top_down).astype(np.uint8)
color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

# Recast the x and y points into usable format for cv2.fillPoly()
pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
pts = np.hstack((pts_left, pts_right))

# Draw the lane onto the warped blank image
cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

# Warp the blank back to original image space using inverse perspective matrix (Minv)
newwarp = transform_perspective(color_warp, (color_warp.shape[1], color_warp.shape[0]), dst, src)
# Combine the result with the original image
print(undistorted.shape, newwarp.shape)
result = cv2.addWeighted(undistorted, 1, newwarp, 0.3, 0)
plt.imshow(result)
