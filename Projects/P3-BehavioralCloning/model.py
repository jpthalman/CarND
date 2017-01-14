"""
Dependencies:
    Numpy
    OpenCV 3
    Pandas
    SciKit-Learn
    Keras
"""

import numpy as np
import utils

# Load the data
images, angles = utils.load_data()
n_obs, im_h, im_w, color_ch = images.shape
print("Loaded %d observations with image shape %r." % (n_obs, (im_h, im_w, color_ch)))

# Split into training and validation sets
X_train, X_val, y_train, y_val = utils.split_data(images, angles, test_size=0.2, shuffle=True)
print('Split original data into %d training samples and %d validation samples.'
      % (y_train.shape[0], y_val.shape[0]))

