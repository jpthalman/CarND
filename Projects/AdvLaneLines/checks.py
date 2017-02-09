"""
The purpose of these functions are to check the validity of the predicted lane lines. They must have a
boolean return and should perform one check per function.
"""

import numpy as np
import cv2


def roughly_parallel(left, right, percent):
    similar = True
    for l_coeff, r_coeff in zip(map(abs, left[:2]), map(abs, right[:2])):
        similar &= abs(l_coeff - r_coeff)/max(l_coeff, r_coeff) < percent
    return similar


def similar_curvature(left, right, percent):
    min = np.min([left, right])
    max = np.max([left, right])
    return min/max > percent if max < 1000 else True
