import os
from collections import namedtuple
import numpy as np
import cv2
import matplotlib.pyplot as plt


def colorspace_threshold(im, color_space, channel, thresholds, clahe=False):
    """
    Returns a binary heatmap of a transformed color channel of an image.

    Transforms the image to the given color space, takes the given color channel,
    and returns a binary heat map of the values between the upper and lower thresholds,
    (lower, upper].

    :param im: The image
    :param color_space: The cv2 color space to transform the image to.
    :param channel: The channel to grab from the new color shape. Should be 0, 1, 2, or None.
    :param thresholds: Tuple containing (lower, upper), both between 0-255
    :param clahe: Whether or not to apply CLAHE
    :return: Binary heat map with same dimensions as the original image.
    """
    new_color_space = cv2.cvtColor(im, color_space)

    if channel is not None:
        color_ch = new_color_space[..., channel]
    else:
        color_ch = new_color_space

    if clahe:
        eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        color_ch = eq.apply(color_ch)

    color_ch = np.uint8(255.*color_ch/np.max(color_ch))
    lower, upper = thresholds

    binary_output = np.zeros_like(color_ch)
    binary_output[(color_ch > lower) & (color_ch <= upper)] = 1
    return binary_output


def gradient_threshold(im, color_space, channel, method, thresholds, kernel_size=3):
    """
    Returns a binary heat map where the given gradient type is between the thresholds.

    :param im: The image
    :param color_space: The cv2 color space to transform the image to.
    :param channel: The channel to grab from the new color shape. Should be 0, 1, 2, or None.
    :param method: Choose one of the below:
        `x`: The gradient in the X direction
        `y`: The gradient in the Y direction
        `m`: The magnitude of the gradients
        `d`: The direction of the gradients
    :param kernel_size: The kernel size to use with Sobel gradient calculations.
    :param thresholds: Tuple containing (lower, upper) thresholds.
    :return: Binary heat map with same dimensions as the original image.
    """
    new_color_space = cv2.cvtColor(im, color_space)

    if channel is not None:
        color_ch = new_color_space[..., channel]
    else:
        color_ch = new_color_space

    sobelx = np.abs(cv2.Sobel(color_ch, cv2.CV_64F, 1, 0, ksize=kernel_size))
    sobely = np.abs(cv2.Sobel(color_ch, cv2.CV_64F, 0, 1, ksize=kernel_size))

    if method == 'x':
        operator = sobelx
    elif method == 'y':
        operator = sobely
    elif method == 'm':
        operator = np.sqrt(sobelx**2 + sobely**2)
    elif method == 'd':
        operator = np.arctan2(sobely, sobelx)
    else:
        raise ValueError("Argument 'method' must be 'x', 'y', 'm', or 'd'.")

    lower, upper = thresholds
    operator = np.uint8(255. * operator / np.max(operator))

    binary_output = np.zeros_like(operator)
    binary_output[(operator > lower) & (operator <= upper)] = 1
    return binary_output


def calibrate_camera(cal_ims_dir, nx, ny):
    path = os.getcwd() + cal_ims_dir
    shape = None
    objpoints = []
    imgpoints = []

    objp = np.zeros((nx*ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    for im_path in os.listdir(path):
        im = cv2.imread(path + im_path)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        shape = gray.shape
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        if ret:
            imgpoints.append(corners)
            objpoints.append(objp)

    ret, M, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        (shape[1], shape[0]),
        None,
        None
      )
    return ret, M, dist, rvecs, tvecs


def undistort_img(im, M, dist):
    return cv2.undistort(im, M, dist, None, M)


def transform_perspective(im, new_size, src, dst, interpolation=cv2.INTER_LINEAR):
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(im, M, new_size, flags=interpolation)


def histogram(input):
    hist = np.sum(input[input.shape[0]//2:, :], axis=0)

    mid = hist.shape[0]//2
    left = np.argmax(hist[:mid])
    right = np.argmax(hist[mid:]) + mid
    return hist, left, right


def get_return_values(coords, f, shift=0):
    a, b, c = f
    return (c + shift) + b*coords + a*coords**2


def sliding_window(warped, n_windows, margin=100, minpix=50):
    h, w = warped.shape

    window_size = h // n_windows
    window_idx = -np.arange(-h, 0, window_size)

    _, leftx_base, rightx_base = histogram(warped[h//2:, :])
    leftx_current, rightx_current = leftx_base, rightx_base

    nonzeroy, nonzerox = warped.nonzero()
    left_lane_inds, right_lane_inds = [], []

    for window in window_idx:
        yhigh = window
        ylow = max(window-window_size, 0)
        xleft_high = leftx_current + margin
        xleft_low = leftx_current - margin
        xright_high = rightx_current + margin
        xright_low = rightx_current - margin

        good_left_inds = (
            (nonzeroy >= ylow) &
            (nonzeroy < yhigh) &
            (nonzerox >= xleft_low) &
            (nonzerox < xleft_high)
          ).nonzero()[0]
        good_right_inds = (
            (nonzeroy >= ylow) &
            (nonzeroy < yhigh) &
            (nonzerox >= xright_low) &
            (nonzerox < xright_high)
          ).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    return np.polyfit(lefty, leftx, 2), np.polyfit(righty, rightx, 2)


def predict_from_margin_around_prev_fit(im, left, right, margin=100):
    nonzero = im.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_lane_inds = (
        (nonzerox > get_return_values(nonzeroy, left, -margin)) &
        (nonzerox < get_return_values(nonzeroy, left, margin))
      )
    right_lane_inds = (
        (nonzerox > get_return_values(nonzeroy, right, -margin)) &
        (nonzerox < get_return_values(nonzeroy, right, margin))
      )

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    return np.polyfit(lefty, leftx, 2), np.polyfit(righty, rightx, 2)
