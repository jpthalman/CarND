import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

from processing import calibrate_camera, undistort_img, colorspace_threshold, gradient_threshold, \
                       transform_perspective, histogram


ret, M, dist, rvecs, tvecs = calibrate_camera('\\camera_cal\\', 9, 6)

test_im = os.getcwd() + '\\test_images\\test4.jpg'
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
    [150, 675],
    [xmax//2 - x_shift, 450],
    [xmax//2 + x_shift, 450],
    [xmax-150, 675]
  ])

shift = 100
dst = np.float32([
    [shift, ymax],
    [shift, shift],
    [xmax-shift, shift],
    [xmax-shift, ymax]
  ])

top_down = transform_perspective(thresh_img, (xmax, ymax), src, dst)
plt.imshow(top_down, cmap='gray')
hist, lmax, rmax = histogram(top_down)
plt.plot(hist)
print(lmax, rmax)


