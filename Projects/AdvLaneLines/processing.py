import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


def binary_colorspace_threshold(im, color_space, channel, thresholds):
    """
    Returns a binary heatmap of a transformed color channel of an image.

    Transforms the image to the given color space, takes the given color channel,
    and returns a binary heatmap of the values between the upper and lower thresholds,
    (lower, upper].

    :param im: The image
    :param color_space: cv2.COLOR_SPACE
    :param channel: 0, 1, 2, or None
    :param thresholds: Tuple containing (lower, upper), both between 0-255
    :return: Binary heat map with same dimensions as the original image.
    """
    new_color_space = cv2.cvtColor(im, color_space)

    if channel is not None:
        color_ch = new_color_space[..., channel]
    else:
        color_ch = new_color_space

    color_ch = np.uint8(255*color_ch/np.max(color_ch))
    lower, upper = thresholds

    binary_output = np.zeros_like(color_ch)
    binary_output[(color_ch > lower) & (color_ch <= color_ch)] = 1
    return binary_output


def sobel_gradient_threshold(im, method, kernel_size, thresholds):
    assert im.ndim == 2, "Argument 'im' must be a single color channel."

    sobelx = np.abs(cv2.Sobel(im, cv2.CV_64F, 1, 0, ksize=kernel_size))
    sobely = np.abs(cv2.Sobel(im, cv2.CV_64F, 0, 1, ksize=kernel_size))

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
    binary_output = np.zeros_like(operator)
    binary_output[(operator > lower) & (operator <= upper)] = 1
    return binary_output


def calibrate_camera(cal_ims_dir, nx, ny):
    cwd = os.getcwd()
    shape = None
    objpoints = []
    imgpoints = []

    objp = np.zeros((nx*ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    for im_path in os.listdir(cwd + cal_ims_dir):
        im = cv2.imread(im_path)
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



