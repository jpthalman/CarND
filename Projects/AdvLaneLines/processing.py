import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


def colorspace_threshold(im, thresholds, color_space=None, channel=None, clahe=False):
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
    if color_space is not None:
        im = cv2.cvtColor(im, color_space)

    if channel is not None:
        color_ch = im[..., channel]
    else:
        color_ch = im

    if clahe:
        eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        color_ch = eq.apply(color_ch)

    color_ch = np.uint8(255.*color_ch/np.max(color_ch))
    lower, upper = thresholds

    binary_output = np.zeros_like(color_ch)
    binary_output[(color_ch > lower) & (color_ch <= upper)] = 1
    return binary_output


def gradient_threshold(im, channel, method, thresholds, color_space=None, kernel_size=3):
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
    if color_space is not None:
        im = cv2.cvtColor(im, color_space)

    if channel is not None:
        color_ch = im[..., channel]
    else:
        color_ch = im

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


def n_bitwise_or(*args):
    """
    Takes in N binary images and combines them using bitwise or.

    :param args: N binary images of the same size
    :return: Combined binary images.
    """
    output = args[0]
    for i in range(1, len(args)):
        output = cv2.bitwise_or(output, args[i])
    return output


def calibrate_camera(cal_ims_dir, nx, ny):
    """
    Uses chessboard images to calibrate a distortion matrix to correct for warping.

    :param cal_ims_dir: The directory which contains the chessboard images
    :param nx: The number of horizontal corners in the chessboards
    :param ny: The number of vertical corners in the chessboards
    :return: The return values of cv2.calibrateCamera
        `ret`: Whether the calibration was successful
        `M`: The calculated calibration matrix
        `dist`: The calculated distortion coefficients
        `rvecs`: Rotation of the camera
        `tvecs`: Translation of the camera
    """
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
    """
    Uses the calibrated Matrix and Distortion coefficients to undistort an image.

    :param im: The image to undistort
    :param M: The calibrated distortion matrix
    :param dist: The Calibrated distortion coefficients
    :return: Undistorted image
    """
    return cv2.undistort(im, M, dist, None, M)


def transform_perspective(im, new_size, src, dst, interpolation=cv2.INTER_LINEAR):
    """
    Meant to transform the perspective of a road image to a Top-Down view.

    :param im: Original image
    :param new_size: The size of the transformed image
    :param src: Source points on the original image
    :param dst: Destination points to transform the source points to in the transformed image
    :param interpolation: How to fill in missing pixels. cv2.INTER_LINEAR is default.
    :return: Transformed image with size `new_size`
    """
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(im, M, new_size, flags=interpolation)


def histogram(input):
    """
    Creates a histogram of the bottom half of an image and finds the maximums of the left and right sides.

    :param input: The original image
    :return: Histogram of the lower half of the image, left maximum index, right maximum index.
    """
    hist = np.sum(input[input.shape[0]//2:, :], axis=0)

    mid = hist.shape[0]//2
    left = np.argmax(hist[:mid])
    right = np.argmax(hist[mid:]) + mid
    return hist, left, right


def get_return_values(coords, f, shift=0):
    """
    Takes in a coordinate set and a second degree polynomial and returns the functions output for the given coords.

    :param coords: The coordinated to evaluate the polynomial on
    :param f: The polynomial weights
    :param shift: Optional constant shift
    :return: 1D array of the functional output of the polynomial for the given coords
    """
    a, b, c = f
    return (c + shift) + b*coords + a*coords**2


def get_curvature(x, y):
    """
    Given a set of X and Y values for a 2nd degree polynomial, finds the curvature of the lines in world space.

    :param x: X coords
    :param y: Y coords
    :return: Curvature in meters
    """
    ym_per_pix = 30/720
    xm_per_pix = 3.7/700
    y_eval = np.max(y)

    a, b, c = np.polyfit(y*ym_per_pix, x*xm_per_pix, 2)
    curvature = (1 + (2*a*y_eval*ym_per_pix + b)**2)**1.5 / abs(2*a)
    return curvature

def gaussian_blur(im, kernel_size):
    """
    Applies gaussian blurring to an image to reduce noise.

    :param im: Image to smooth
    :param kernel_size: Kernel size for smoothing
    :return: Smoothed image
    """
    return cv2.GaussianBlur(im , (kernel_size, kernel_size), 0)

def sliding_window(warped, n_windows, margin=100, minpix=50):
    """
    Applies the sliding window detection algorithm to an image to detect the lane lines.

    :param warped: Warped image
    :param n_windows: The number of sliding windows to use
    :param margin: The width of each sliding window
    :param minpix: Minimum number of activated pixels per window
    :return: 2nd degree polynomial fit and it's curvature for each lane line
    """
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

    l_curv, r_curv = get_curvature(leftx, lefty), get_curvature(rightx, righty)
    return np.polyfit(lefty, leftx, 2), np.polyfit(righty, rightx, 2), l_curv, r_curv


def predict_from_margin_around_prev_fit(im, left, right, margin=100):
    """
    Uses a margin the previous frames lane lines to make a prediction about this frames lines.

    :param im: Warped image
    :param left: Previous frames left lane line prediciton
    :param right: Previous frames right lane line prediciton
    :param margin: Margin around each line to grab pixels from
    :return: 2nd degree polynomial fit and it's curvature for each lane line
    """
    # Handles the first frame where the is no previous fit
    if left is None or right is None:
        return None, None, None, None

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

    l_curv, r_curv = get_curvature(leftx, lefty), get_curvature(rightx, righty)
    return np.polyfit(lefty, leftx, 2), np.polyfit(righty, rightx, 2), l_curv, r_curv
