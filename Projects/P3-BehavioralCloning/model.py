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

from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Input, Dropout, Flatten
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras.layers.advanced_activations import ELU, PReLU

# Load the data
images, angles = utils.load_data("~/sharefolder/CarND/Projects/P3-BehavioralCloning/Data/driving_log.csv")
n_obs, im_h, im_w, color_ch = images.shape
print("Loaded %d observations with image shape %r." % (n_obs, (im_h, im_w, color_ch)))

# Split into training and validation sets
X_train, X_val, y_train, y_val = utils.split_data(images, angles, test_size=0.2, shuffle=True)
print('Split original data into %d training samples and %d validation samples.'
      % (y_train.shape[0], y_val.shape[0]))

X_train, y_train = utils.flip_and_concat(X_train, y_train)
X_val, y_val = utils.flip_and_concat(X_val, y_val)


# Model construction
model = Sequential()

model.add(BatchNormalization(mode=0, axis=4, input_shape=(160, 320, 3)))

model.add(Convolution2D(24, 5, 5, subsample=(2,2), border_mode='valid', activation=ELU()))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), border_mode='valid', activation=ELU()))
model.add(Convolution2D(48, 3, 3, border_mode='valid', activation=ELU()))
model.add(Convolution2D(64, 3, 3, border_mode='valid', activation=ELU()))
model.add(MaxPooling2D(pool_size=(2,3), border_mode='valid'))

model.add(Flatten())

model.add(Dense(100, activation=PReLU()))
model.add(Dense(50, activation=PReLU()))
model.add(Dense(10, activation=PReLU()))
model.add(Dense(1))


# Model training
model.compile('Nadam', 'mse', ['accuracy'])
model.fit(X_train, y_train, batch_size=64, nb_epoch=10, validation_data=(X_val, y_val))


# Save model weights and structure
with open('model.json', 'w') as file:
    file.write(model.to_json())
model.save_weights('model.h5')