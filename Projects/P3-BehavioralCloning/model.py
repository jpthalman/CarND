import numpy as np
from sklearn.model_selection import train_test_split
import utils

# Load the data
images, angles = utils.load_data()
n_obs, im_h, im_w, color_ch = images.shape
print("Loaded %d observations with image shape %r." % (n_obs, (im_h, im_w, color_ch)))

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, angles, test_size=0.2)

