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
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras.layers.advanced_activations import ELU
from keras.optimizers import adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

import utils


__author__ = 'Jacob Thalman'
__email__ = 'jpthalman@gmail.com'


# Create hyper-parameters
Parameters = namedtuple('Parameters', [
    # General settings
    'batch_size', 'max_epochs', 'path', 'angle_shift',
    # Optimizer settings
    'learning_rate', 'epsilon', 'decay',
    # Training settings
    'min_delta', 'patience', 'kwargs'
  ])

params = Parameters(
    # General settings
    batch_size=64, max_epochs=100, path='', angle_shift=0.1,
    # Optimizer settings
    learning_rate=1e-3, epsilon=1e-8, decay=0.0,
    # Training settings
    min_delta=1e-4, patience=10, kwargs={'prob': 0.8}
  )


path = '/home/japata/sharefolder/CarND/Projects/BehavioralCloning/Data/'

# Load the data from the middle runs
center = utils.load_data(path + 'Center/driving_log.csv')

# Remove 90% of the frames where the steering angle is close to zero
center_ims, center_angs = utils.keep_n_percent_of_data_where(
    data=center['center'],
    values=center['angles'],
    condition_lambda=lambda x: abs(x) < 1e-5,
    percent=0.1
  )

# Load the data from the left runs
left = utils.load_data(path + 'Left/driving_log.csv')

# Remove 100% of the frames where the steering angle is less than 1e-5
# This effectively only keeps the frames where the car is recovering from
# the left edge.
left_ims, left_angs = utils.keep_n_percent_of_data_where(
    data=left['center'],
    values=left['angles'],
    condition_lambda=lambda x: x < 1e-5,
    percent=0.0
  )

# Load the data from the left runs
right = utils.load_data(path + 'Right/driving_log.csv')

# Remove 100% of the frames where the steering angle is greater than -1e-5
# This effectively only keeps the frames where the car is recovering from
# the right edge.
right_ims, right_angs = utils.keep_n_percent_of_data_where(
    data=right['center'],
    values=right['angles'],
    condition_lambda=lambda x: x > -1e-5,
    percent=0.0
  )

# Aggregate all sets into one.
filtered_images = np.concatenate((center_ims, right_ims, left_ims), axis=0)
filtered_angles = np.concatenate((center_angs, 0.9*right_angs, 0.9*left_angs), axis=0)

# Split the data into training and validation sets
train_paths, val_paths, train_angles, val_angles = utils.split_data(
    features=filtered_images,
    labels=filtered_angles,
    test_size=0.2,
    shuffle_return=True
  )
print('Training size: %d | Validation size: %d' % (train_paths.shape[0], val_paths.shape[0]))


# Model construction
model = Sequential([
    Convolution2D(3, 1, 1, border_mode='valid', init='he_normal', input_shape=(66, 200, 1)),

    Convolution2D(24, 5, 5, border_mode='valid', init='he_normal', input_shape=(66, 200, 3)),
    MaxPooling2D(pool_size=(2,2), border_mode='valid'),
    ELU(),

    Convolution2D(36, 5, 5, border_mode='valid', init='he_normal', input_shape=(31, 98, 24)),
    MaxPooling2D(pool_size=(2, 2), border_mode='same'),
    ELU(),

    Convolution2D(48, 5, 5, border_mode='valid', init='he_normal', input_shape=(14, 47, 36)),
    MaxPooling2D(pool_size=(2, 2), border_mode='same'),
    ELU(),

    Convolution2D(64, 3, 3, border_mode='valid', init='he_normal', input_shape=(5, 22, 48)),
    ELU(),

    Convolution2D(64, 3, 3, border_mode='valid', init='he_normal', input_shape=(3, 20, 64)),
    ELU(),

    Flatten(),

    Dense(100, init='he_normal'),
    ELU(),

    Dense(50, init='he_normal'),
    ELU(),

    Dense(10, init='he_normal'),
    ELU(),

    Dense(1)
])

optimizer = adam(
    lr=params.learning_rate,
    epsilon=params.epsilon,
    decay=params.decay
  )
model.compile(optimizer=optimizer, loss='mse')

print('\n', model.summary(), '\n')

# Save model structure
with open('model.json', 'w') as file:
    file.write(model.to_json())


# Model training
callbacks = [
    EarlyStopping(monitor='val_loss', min_delta=params.min_delta, patience=params.patience,
                  mode='min'),
    ModelCheckpoint(filepath='model.h5', monitor='val_loss', save_best_only=True,
                    save_weights_only=True, mode='min'),
    TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True)
  ]
model.fit_generator(
    generator=utils.batch_generator(ims=train_paths, angs=train_angles, batch_size=params.batch_size,
                                    augmentor=utils.augment_image, path=params.path, kwargs=params.kwargs),
    samples_per_epoch=400*params.batch_size,
    nb_epoch=params.max_epochs,
    # Halve the batch size, as `utils.val_augmentor` doubles the batch size by flipping the images and angles
    validation_data=utils.batch_generator(ims=val_paths, angs=val_angles, batch_size=params.batch_size//2,
                                          augmentor=utils.val_augmentor, validation=True, path=params.path),
    # Double the size of the validation set, as we are flipping the images to balance the right/left
    # distribution.
    nb_val_samples=2*val_paths.shape[0],
    callbacks=callbacks
  )
