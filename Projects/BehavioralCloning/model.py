"""
Dependencies:
    Keras
    Numpy
    OpenCV 3
    Pandas
    SciKit-Learn
"""

import numpy as np
import os
from collections import namedtuple
from glob import glob
from sklearn.utils import shuffle
import utils
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras.regularizers import l2
from keras.layers import Activation
from keras.layers.advanced_activations import ELU, PReLU
from keras.optimizers import adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard


__author__ = 'Jacob Thalman'
__email__ = 'jpthalman@gmail.com'

# TODO:
# Use discarded straight paths from the 'center' images as a validation set.
# This will force the model to learn from images in which it has to turn,
# but it will be evaluated on how well it can go straight.


# Create hyper-parameters
Parameters = namedtuple('Parameters', [
    # General settings
    'batch_size', 'max_epochs', 'path', 'angle_shift',
    # Model settings
    'l2_reg', 'keep_prob',
    # Optimizer settings
    'learning_rate', 'epsilon', 'decay',
    # Training settings
    'min_delta', 'patience', 'kwargs'
  ])

params = Parameters(
    # General settings
    batch_size=256, max_epochs=100, path='', angle_shift=0.15,
    # Model settings
    l2_reg=0.0005, keep_prob=0.5,
    # Optimizer settings
    learning_rate=1e-3, epsilon=1e-8, decay=0.0,
    # Training settings
    min_delta=1e-4, patience=9, kwargs={'prob': 0.9}
  )


path = '/home/japata/sharefolder/CarND/Projects/BehavioralCloning/'

# Load Udacity's Data
udacity_paths, udacity_angs = utils.concat_all_cameras(
    data=utils.load_data(path + 'UdacityData/', 'driving_log.csv'),
    angle_shift=params.angle_shift,
    condition_lambda=lambda x: abs(x) < 1e-5,
    keep_percent=0.4
  )

# Load the data from the middle runs
# Remove 90% of the frames where the steering angle is close to zero
center_paths, center_angs = utils.concat_all_cameras(
    data=utils.load_data(path + 'Data/Center/', 'driving_log.csv'),
    angle_shift=params.angle_shift,
    condition_lambda=lambda x: abs(x) < 1e-5,
    keep_percent=0.4
  )

# Load the data from the left runs
# Remove 100% of the frames where the steering angle is less than 1e-5
# This effectively only keeps the frames where the car is recovering from
# the left edge.
left_paths, left_angs = utils.concat_all_cameras(
    data=utils.load_data(path + 'Data/Left/', 'driving_log.csv'),
    angle_shift=params.angle_shift,
    condition_lambda=lambda x: x < 1e-5,
    keep_percent=0.0,
    drop_camera='right'
  )

# Load the data from the right runs
# Remove 100% of the frames where the steering angle is greater than -1e-5
# This effectively only keeps the frames where the car is recovering from
# the right edge.
right_paths, right_angs = utils.concat_all_cameras(
    data=utils.load_data(path + 'Data/Right/', 'driving_log.csv'),
    angle_shift=params.angle_shift,
    condition_lambda=lambda x: x > -1e-5,
    keep_percent=0.0,
    drop_camera='left'
  )

# Aggregate all sets into one.
filtered_paths = np.concatenate((udacity_paths, center_paths, right_paths, left_paths), axis=0)
filtered_angs = np.concatenate((udacity_angs, center_angs, right_angs, left_angs), axis=0)

train_paths, val_paths, train_angs, val_angs = utils.split_data(
    features=udacity_paths,
    labels=udacity_angs,
    test_size=0.05
  )

print('Training size: %d | Validation size: %d' % (train_paths.shape[0], val_paths.shape[0]))


# Model construction
model = Sequential([
    Lambda(lambda x: x/255. - 0.5, input_shape=(66, 200, 3)),

    Convolution2D(3, 1, 1, border_mode='valid'),

    # 66x200x3
    Convolution2D(24, 5, 5, border_mode='valid', activation='elu', W_regularizer=l2(params.l2_reg), subsample=(2, 2)),
    Dropout(params.keep_prob),
    # 31x98x24
    Convolution2D(36, 5, 5, border_mode='valid', activation='elu', W_regularizer=l2(params.l2_reg), subsample=(2, 2)),
    Dropout(params.keep_prob),
    # 14x47x36
    Convolution2D(48, 5, 5, border_mode='valid', activation='elu', W_regularizer=l2(params.l2_reg), subsample=(2, 2)),
    Dropout(params.keep_prob),
    # 5x22x48
    Convolution2D(64, 3, 3, border_mode='valid', activation='elu', W_regularizer=l2(params.l2_reg)),
    Dropout(params.keep_prob),
    # 3x20x64
    Convolution2D(64, 3, 3, border_mode='valid', activation='elu', W_regularizer=l2(params.l2_reg)),
    Dropout(params.keep_prob),

    # 1x18x64
    Flatten(),

    Dense(128, activation='elu', W_regularizer=l2(params.l2_reg)),
    Dropout(params.keep_prob),

    Dense(64, activation='elu', W_regularizer=l2(params.l2_reg)),
    Dropout(params.keep_prob),

    Dense(16, activation='elu', W_regularizer=l2(params.l2_reg)),

    Dense(1)
  ])

optimizer = adam(
    lr=params.learning_rate,
    epsilon=params.epsilon,
    decay=params.decay
  )
model.compile(optimizer=optimizer, loss='mse')

print('\n', model.summary(), '\n')


# Clear TensorBoard Logs
for file in os.listdir('./logs/'):
    os.remove('./logs/' + file)

# Remove previous model files
try:
    for file in glob('Models/*.h5'):
        os.remove(file)
    os.remove('model.json')
    os.remove('model.h5')
except FileNotFoundError:
    pass


# Save model structure
with open('model.json', 'w') as file:
    file.write(model.to_json())


# Model training
filepath = 'Models/model_{epoch:03d}_{val_loss:0.5f}.h5'
callbacks = [
    EarlyStopping(monitor='val_loss', min_delta=params.min_delta, patience=params.patience,
                  mode='min'),
    ModelCheckpoint(filepath=filepath, monitor='val_loss', save_best_only=True,
                    save_weights_only=True, mode='min'),
    TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True)
  ]

model.fit_generator(
    generator=utils.batch_generator(ims=train_paths, angs=train_angs, batch_size=params.batch_size,
                                    augmentor=utils.augment_image, path=params.path, kwargs=params.kwargs),
    samples_per_epoch=25600,
    nb_epoch=params.max_epochs,
    # Halve the batch size, as `utils.val_augmentor` doubles the batch size by flipping the images and angles
    validation_data=utils.batch_generator(ims=val_paths, angs=val_angs, batch_size=params.batch_size//2,
                                          augmentor=utils.val_augmentor, validation=True, path=params.path),
    # Double the size of the validation set, as we are flipping the images to balance the right/left
    # distribution.
    nb_val_samples=2*udacity_angs.shape[0],
    callbacks=callbacks
  )

