"""
Dependencies:
    Keras
    Numpy
    OpenCV 3
    Pandas
    SciKit-Learn
"""


from collections import namedtuple
import numpy as np
import os

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Lambda
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras.regularizers import l2
from keras.layers.advanced_activations import ELU, PReLU
from keras.layers import Activation
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
    'patience', 'kwargs'
  ])

params = Parameters(
    # General settings
    batch_size=64, max_epochs=100, path='', angle_shift=0.15,
    # Model settings
    l2_reg=0.0, keep_prob=0.5,
    # Optimizer settings
    learning_rate=1e-3, epsilon=1e-8, decay=0.0,
    # Training settings
    patience=10, kwargs={'prob': 0.75}
  )


# Load Udacity's Data
udacity_paths, udacity_angs = utils.concat_all_cameras(
    data=utils.load_data('/home/japata/sharefolder/CarND/Projects/BehavioralCloning/UdacityData/', 'driving_log.csv'),
    angle_shift=params.angle_shift,
    condition_lambda=lambda x: abs(x) < 1e-5,
    keep_percent=0.15
  )


path = '/home/japata/sharefolder/CarND/Projects/BehavioralCloning/Data/'

# Load the data from the middle runs
# Remove 90% of the frames where the steering angle is close to zero
center_ims, center_angs = utils.concat_all_cameras(
    data=utils.load_data(path + 'Center/', 'driving_log.csv'),
    angle_shift=params.angle_shift,
    condition_lambda=lambda x: abs(x) < 1e-5,
    keep_percent=0.15
  )

# Load the data from the left runs
# Remove 100% of the frames where the steering angle is less than 1e-5
# This effectively only keeps the frames where the car is recovering from
# the left edge.
left_ims, left_angs = utils.concat_all_cameras(
    data=utils.load_data(path + 'Left/', 'driving_log.csv'),
    angle_shift=params.angle_shift,
    condition_lambda=lambda x: x < 1e-5,
    keep_percent=0.0
  )

# Load the data from the right runs
# Remove 100% of the frames where the steering angle is greater than -1e-5
# This effectively only keeps the frames where the car is recovering from
# the right edge.
right_ims, right_angs = utils.concat_all_cameras(
    data=utils.load_data(path + 'Right/', 'driving_log.csv'),
    angle_shift=params.angle_shift,
    condition_lambda=lambda x: x > -1e-5,
    keep_percent=0.0
  )

# Aggregate all sets into one.
filtered_paths = np.concatenate((udacity_paths, center_ims, right_ims, left_ims), axis=0)
filtered_angs = np.concatenate((udacity_angs, center_angs, right_angs, left_angs), axis=0)

# Split the data into training and validation sets
train_paths, val_paths, train_angles, val_angles = utils.split_data(
    features=filtered_paths,
    labels=filtered_angs,
    test_size=0.2,
    shuffle_return=True
  )

val_size = val_paths.shape[0]
# Set the minimum change in validation accuracy to the resolution of the validation set
min_delta = 30 / (2*val_size)

print('Training size: %d | Validation size: %d | Min Delta: %04.4f'
      % (train_paths.shape[0], val_size, min_delta))




# Model construction
model = Sequential([
    Lambda(lambda x: x/255. - 0.5, input_shape=(66, 200, 1)),

    Convolution2D(3, 1, 1, border_mode='valid', init='he_normal'),

    Convolution2D(24, 5, 5, border_mode='valid', init='he_normal'),
    # 62x194x24
    MaxPooling2D(pool_size=(2,2), border_mode='valid'),
    Activation('relu'),

    # 31x98x24
    Convolution2D(36, 5, 5, border_mode='valid', init='he_normal'),
    # 27x94x36
    MaxPooling2D(pool_size=(2,2), border_mode='same'),
    Activation('relu'),

    # 13x47x36
    Convolution2D(48, 5, 5, border_mode='valid', init='he_normal'),
    # 9x43x48
    MaxPooling2D(pool_size=(2,2), border_mode='same'),
    Activation('relu'),

    # 5x22x48
    Convolution2D(64, 3, 3, border_mode='valid', init='he_normal'),
    Activation('relu'),

    # 3x20x64
    Convolution2D(64, 3, 3, border_mode='valid', init='he_normal'),
    Activation('relu'),

    # 1x18x64
    Flatten(),

    Dense(100, init='he_normal'),
    Activation('relu'),

    Dense(50, init='he_normal'),
    Activation('relu'),

    Dense(10, init='he_normal'),
    Activation('relu'),

    Dense(1, init='he_normal')
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
    os.remove('model.json')
    os.remove('model.h5')
except FileNotFoundError:
    pass


# Model training
callbacks = [
    EarlyStopping(monitor='val_loss', min_delta=min_delta, patience=params.patience,
                  mode='min'),
    ModelCheckpoint(filepath='model.h5', monitor='val_loss', save_best_only=True,
                    save_weights_only=True, mode='min'),
    TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True)
  ]
model.fit_generator(
    generator=utils.batch_generator(ims=train_paths, angs=train_angles, batch_size=params.batch_size,
                                    augmentor=utils.augment_image, path=params.path, kwargs=params.kwargs),
    samples_per_epoch=800*params.batch_size,
    nb_epoch=params.max_epochs,
    # Halve the batch size, as `utils.val_augmentor` doubles the batch size by flipping the images and angles
    validation_data=utils.batch_generator(ims=val_paths, angs=val_angles, batch_size=params.batch_size//2,
                                          augmentor=utils.val_augmentor, validation=True, path=params.path),
    # Double the size of the validation set, as we are flipping the images to balance the right/left
    # distribution.
    nb_val_samples=2*val_paths.shape[0],
    callbacks=callbacks
  )

# Save model structure
with open('model.json', 'w') as file:
    file.write(model.to_json())