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

import utils


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
    batch_size=128, max_epochs=100, path='', angle_shift=0.15,
    # Model settings
    l2_reg=1e-4, keep_prob=0.2,
    # Optimizer settings
    learning_rate=1e-3, epsilon=1e-8, decay=0.0,
    # Training settings
    min_delta=1e-4, patience=4, kwargs={'prob': 1.0}
  )


path = '/home/japata/sharefolder/CarND/Projects/BehavioralCloning/'

# Load Udacity's Data
udacity_paths, udacity_angs = utils.concat_all_cameras(
    data=utils.load_data(path + 'UdacityData/', 'driving_log.csv'),
    angle_shift=params.angle_shift,
    condition_lambda=lambda x: abs(x) < 1e-5,
    keep_percent=0.8
  )

# Load the data from the middle runs
# Remove 90% of the frames where the steering angle is close to zero
center_paths, center_angs = utils.concat_all_cameras(
    data=utils.load_data(path + 'Data/Center/', 'driving_log.csv'),
    angle_shift=params.angle_shift,
    condition_lambda=lambda x: abs(x) < 1e-5,
    keep_percent=1.0
  )

# Load the data from the left runs
# Remove 100% of the frames where the steering angle is less than 1e-5
# This effectively only keeps the frames where the car is recovering from
# the left edge.
left_paths, left_angs = utils.concat_all_cameras(
    data=utils.load_data(path + 'Data/Left/', 'driving_log.csv'),
    angle_shift=params.angle_shift,
    condition_lambda=lambda x: np.array([True if i < 1e-5 or i > 0.9 else False for i in x]),
    keep_percent=0.0
  )

# Load the data from the right runs
# Remove 100% of the frames where the steering angle is greater than -1e-5
# This effectively only keeps the frames where the car is recovering from
# the right edge.
right_paths, right_angs = utils.concat_all_cameras(
    data=utils.load_data(path + 'Data/Right/', 'driving_log.csv'),
    angle_shift=params.angle_shift,
    condition_lambda=lambda x: np.array([True if i > -1e-5 or i < -0.9 else False for i in x]),
    keep_percent=0.0
  )

# Aggregate all sets into one.
filtered_paths = np.concatenate((center_paths, right_paths, left_paths), axis=0)
filtered_angs = np.concatenate((center_angs, right_angs, left_angs), axis=0)

train_paths, train_angs = shuffle(filtered_paths, filtered_angs)

print('Training size: %d | Validation size: %d' % (train_paths.shape[0], udacity_paths.shape[0]))


# Model construction
model = Sequential([
    Lambda(lambda x: x/255. - 0.5, input_shape=(66, 200, 3)),

    Convolution2D(24, 5, 5, subsample=(2,2), W_regularizer=l2(params.l2_reg)),
    ELU(),
    Dropout(params.keep_prob),

    Convolution2D(36, 5, 5, subsample=(2,2), W_regularizer=l2(params.l2_reg)),
    ELU(),
    Dropout(params.keep_prob),

    Convolution2D(48, 5, 5, subsample=(2,2), W_regularizer=l2(params.l2_reg)),
    ELU(),
    Dropout(params.keep_prob),

    Convolution2D(64, 3, 3, W_regularizer=l2(params.l2_reg)),
    ELU(),
    Dropout(params.keep_prob),

    Convolution2D(64, 3, 3, W_regularizer=l2(params.l2_reg)),
    ELU(),
    Dropout(params.keep_prob),

    Flatten(),

    Dense(100, W_regularizer=l2(params.l2_reg)),
    ELU(),
    Dropout(params.keep_prob),

    Dense(10, W_regularizer=l2(params.l2_reg)),
    ELU(),

    Dense(1, W_regularizer=l2(params.l2_reg))
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
    validation_data=utils.batch_generator(ims=udacity_paths, angs=udacity_angs, batch_size=params.batch_size//2,
                                          augmentor=utils.val_augmentor, validation=True, path=params.path),
    # Double the size of the validation set, as we are flipping the images to balance the right/left
    # distribution.
    nb_val_samples=2*udacity_angs.shape[0],
    callbacks=callbacks
  )
