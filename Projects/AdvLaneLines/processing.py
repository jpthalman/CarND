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



