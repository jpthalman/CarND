"""
The purpose of these functions are to check the validity of the predicted lane lines. They must have a
boolean return and should perform one check per function.
"""

import numpy as np
import cv2


def roughly_parallel(left, right, percent):
    """
    If the two curves non-linear coefficients are similar enough, return true.

    :param left: Left curve
    :param right: Right curve
    :param percent: Maximum divergence of the lines
    :return: Boolean
    """
    similar = True
    for l_coeff, r_coeff in zip(map(abs, left[:2]), map(abs, right[:2])):
        similar &= abs(l_coeff - r_coeff)/max(l_coeff, r_coeff) < percent
    return similar


def similar_curvature(left, right, percent):
    """
    Measures the similarity of the two curvatures.

    :param left: Left curvature
    :param right: Right curvature
    :param percent: Max percent divergence
    :return: Boolean
    """
    min = np.min([left, right])
    max = np.max([left, right])
    return min/max > percent if max < 1000 else True


def not_same_line(left, right):
    """
    Ensures that the two lane lines are not the same line, indicating failure.

    :param left: Left line
    :param right: Right line
    :return: Boolean
    """
    return abs(left[2] - right[2]) > 1e-2
