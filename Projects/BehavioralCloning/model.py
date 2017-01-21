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
    'batch_size', 'max_epochs', 'angle_shift',
    # Optimizer settings
    'learning_rate', 'epsilon', 'decay',
    # Training settings
    'min_delta', 'patience', 'kwargs'
  ])

params = Parameters(
    # General settings
    batch_size=64, max_epochs=100, angle_shift=0.1,
    # Optimizer settings
    learning_rate=1e-3, epsilon=1e-8, decay=0.0,
    # Training settings
    min_delta=1e-4, patience=10, kwargs={'prob': 0.8}
  )


# Load the data
path = '/home/japata/sharefolder/CarND/Projects/BehavioralCloning/Data/'
data = utils.load_data(path + 'driving_log.csv')

# Remove 90% of the frames where the steering angle is close to zero
ims, angles = utils.keep_n_percent_of_data_where(
    data=np.array([data['center'], data['right'], data['left']]).T,
    values=data['angles'],
    condition_lambda=lambda x: abs(x) < 1e-5,
    percent=0.1
  )
center_ims, right_ims, left_ims = ims[..., 0], ims[..., 1], ims[..., 2]

# Modify the steering angles of the left and right cameras's images to simulate
# steering back towards the middle. Aggregate all sets into one.
filtered_images = np.concatenate((center_ims, right_ims, left_ims), axis=0)
filtered_angles = np.concatenate((angles, angles + params.angle_shift, angles - params.angle_shift), axis=0)

# Split the data into training and validation sets
train_paths, val_paths, train_angles, val_angles = utils.split_data(
    features=filtered_images,
    labels=filtered_angles,
    test_size=0.2,
    shuffle_return=True
  )


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
                                    augmentor=utils.augment_image, kwargs=params.kwargs, path=path),
    samples_per_epoch=400*params.batch_size,
    nb_epoch=params.max_epochs,
    # Halve the batch size, as `utils.val_augmentor` doubles the batch size by flipping the images and angles
    validation_data=utils.batch_generator(ims=val_paths, angs=val_angles, batch_size=params.batch_size//2,
                                          augmentor=utils.val_augmentor, path=path, validation=True),
    # Double the size of the validation set, as we are flipping the images to balance the right/left
    # distribution.
    nb_val_samples=2*val_paths.shape[0],
    callbacks=callbacks
  )
