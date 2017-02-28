import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sklearn
import pdb
from skimage.feature import hog


# Define a function to return HOG features and visualization
def get_hog_features(img, orient=9, pix_per_cell=8, cell_per_block=2,
                     vis=False, feature_vec=True):
    """
    Takes in a color channel of an image and returns the HOG features and optionally a visualization.

    :param img: Single color channel image
    :param orient: Number of bins to group the gradients for each cell
    :param pix_per_cell: Number of pixels per cell
    :param cell_per_block: Number of cells ber block
    :param vis: Boolean. Returns `features, hog_image` instead of only `features` if true.
    :param feature_vec: Boolean. Whether or not to flatten the HOG features.
    """
    if vis:
        features, hog_image = hog(
            img,
            orientations=orient,
            pixels_per_cell=(pix_per_cell, pix_per_cell),
            cells_per_block=(cell_per_block, cell_per_block),
            transform_sqrt=True,
            visualise=vis,
            feature_vector=feature_vec)
        return features, hog_image
    else:
        features = hog(
            img,
            orientations=orient,
            pixels_per_cell=(pix_per_cell, pix_per_cell),
            cells_per_block=(cell_per_block, cell_per_block),
            transform_sqrt=True,
            visualise=vis,
            feature_vector=feature_vec)
        return features


# Define a function to compute binned color features
def bin_spatial(img, size=(16, 16)):
    """
    Resize the image and return the color values for each pixel in a flattened array.

    :param img: Image to analyze
    :param size: Tuple with values indicating shape to resize image to
    """
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features


# Define a function to compute color histogram features
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32):
    """
    Compute the histogram of color intensity for each channel of the image.

    :param img: Input image
    :param nbins: Number of histogram bins for each image
    :return: Array of len 3*nbins
    """
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


def single_img_features(img,
                        hog_img=None,
                        color_space='YCrCb',
                        spatial_size=(32, 32),
                        hist_bins=32,
                        orient=9,
                        pix_per_cell=8,
                        cell_per_block=2,
                        hog_channel='ALL',
                        spatial_feat=True,
                        hist_feat=True,
                        hog_feat=True):
    """
    Function to apply all feature analyses and concatenate their results.

    :param img: Input image
    :param hog_img: Precomputed gradients for input image
    :param color_space: Color space to transform the image to
    :param spatial_size: Size transform tuple for the spatial features
    :param hist_bins: Number of histogram bins for color_hist
    :param orient: Number of gradient bins for HOG features
    :param pix_per_cell: For HOG features
    :param cell_per_block: For HOG features
    :param hog_channel: Channel number to perform HOG analysis on. Can be 'ALL'.
    :param spatial_feat: Boolean.
    :param hist_feat: Boolean.
    :param hog_feat: Boolean.
    :return:
    """
    # 1) Define an empty list to receive features
    img_features = []

    # 2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    else:
        feature_image = np.copy(img)

    # 3) Compute spatial features if flag is set
    if spatial_feat:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # 4) Append features to list
        img_features.append(spatial_features)
    # 5) Compute histogram features if flag is set
    if hist_feat:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        # 6) Append features to list
        img_features.append(hist_features)
    # 7) Compute HOG features if flag is set
    if hog_feat:
        if hog_img is not None:
            hog_features = []
            for channel in range(hog_img.shape[2]):
                hog_features.extend(hog_img[..., channel].ravel())
        elif hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[..., channel],
                                    orient, pix_per_cell, cell_per_block,
                                    vis=False, feature_vec=True))
        else:
            hog_features = get_hog_features(feature_image[..., hog_channel],
                                            orient,
                                            pix_per_cell,
                                            cell_per_block,
                                            vis=False,
                                            feature_vec=True)

        # 8) Append features to list
        img_features.append(hog_features)

    # 9) Return concatenated array of features
    return np.concatenate(img_features)


def extract_features(imgs):
    """
    Extract features from an array of images.
    """
    features = []
    for file in imgs:
        image = cv2.imread(file)
        image = image.astype(np.float32) / 255
        features.append(single_img_features(image))
    return features


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    """
    Helper function to draw regions of interest onto an image.

    :param img: Image to draw on
    :param bboxes: List of ROI's to draw
    :param color: Color to make the boxes on the image
    :param thick: Thickness of the boxes to draw
    :return: Annotated image
    """
    imcopy = np.copy(img)
    for bbox in bboxes:
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    return imcopy
