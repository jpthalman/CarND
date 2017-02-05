import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


def calibrate_camera():
    cwd = os.getcwd()
    pass


def colorspace_threshold(im, color_space, channel, thresholds):
    """
    Returns a binary heatmap of a transformed color channel of an image.

    Transforms the image to the given color space, takes the given color channel,
    and returns a binary heat map of the values between the upper and lower thresholds,
    (lower, upper].

    :param im: The image
    :param color_space: The cv2 color space to transform the image to.
    :param channel: The channel to grab from the new color shape. Should be 0, 1, 2, or None.
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
    binary_output = np.zeros_like(operator)
    binary_output[(operator > lower) & (operator <= upper)] = 1
    return binary_output



