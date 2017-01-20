"""
Dependencies:
    Numpy
    OpenCV 3
    Pandas
    SciKit-Learn
    Keras
"""
import numpy as np
from collections import namedtuple

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras.layers.advanced_activations import ELU
from keras.optimizers import adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

import utils


# Create hyper-parameters
Parameters = namedtuple('Parameters', [
    # General settings
    'batch_size', 'max_epochs', 'angle_shift',
    # Optimizer settings
    'learning_rate', 'epsilon', 'decay',
    # Training settings
    'min_delta', 'patience', 'args'
  ])

params = Parameters(
    # General settings
    batch_size=64, max_epochs=100, angle_shift=0.05,
    # Optimizer settings
    learning_rate=1e-4, epsilon=0.1, decay=0.0,
    # Training settings
    min_delta=0.0, patience=10, args={'prob': 0.8}
  )


# Load the data
path = '/home/japata/sharefolder/CarND/Projects/BehavioralCloning/Data/'
data = utils.load_data(path + 'driving_log.csv')

ims, angles = utils.keep_n_percent_of_data_where(
    data=np.array([data['center'], data['right'], data['left']]).T,
    values=data['angles'],
    condition_lambda=lambda x: abs(x) < 1e-5,
    percent=0.1
  )
center_ims, right_ims, left_ims = ims[..., 0], ims[..., 1], ims[..., 2]

filtered_images = np.concatenate((center_ims, right_ims, left_ims), axis=0)
filtered_angles = np.concatenate((angles, angles + params.angle_shift, angles - params.angle_shift), axis=0)

train_paths, val_paths, train_angles, val_angles = utils.split_data(
    features=filtered_images,
    labels=filtered_angles,
    test_size=0.2,
    shuffle_return=True
  )


# Model construction
model = Sequential([
    Convolution2D(3, 1, 1, border_mode='valid', init='he_normal', input_shape=(160, 320, 1)),

    Convolution2D(24, 8, 8, border_mode='valid', init='he_normal'),
    MaxPooling2D(pool_size=(2,2), border_mode='valid'),
    Dropout(0.5),
    ELU(),

    Convolution2D(36, 5, 5, border_mode='valid', init='he_normal'),
    MaxPooling2D(pool_size=(2, 2), border_mode='valid'),
    Dropout(0.5),
    ELU(),

    Convolution2D(48, 5, 5, border_mode='valid', init='he_normal'),
    MaxPooling2D(pool_size=(2, 2), border_mode='valid'),
    Dropout(0.5),
    ELU(),

    Convolution2D(64, 3, 3, border_mode='valid', init='he_normal'),
    MaxPooling2D(pool_size=(2, 2), border_mode='valid'),
    Dropout(0.5),
    ELU(),

    Convolution2D(64, 2, 2, border_mode='valid', init='he_normal'),
    MaxPooling2D(pool_size=(2, 2), border_mode='valid'),
    Dropout(0.5),
    ELU(),

    Flatten(),

    Dense(1024, init='he_normal'),
    Dropout(0.5),
    ELU(),

    Dense(512, init='he_normal'),
    Dropout(0.5),
    ELU(),

    Dense(256, init='he_normal'),
    Dropout(0.5),
    ELU(),

    Dense(128, init='he_normal'),
    Dropout(0.5),
    ELU(),

    Dense(32, init='he_normal'),
    Dropout(0.5),
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
batches = utils.BatchGenerator(
    batch_size=params.batch_size,
    load=True,
    path=path
  )
callbacks = [
    EarlyStopping(monitor='val_loss', min_delta=params.min_delta, patience=params.patience,
                  mode='min'),
    ModelCheckpoint(filepath='model.h5', monitor='val_loss', save_best_only=True,
                    save_weights_only=True, mode='min'),
    TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True)
  ]
model.fit_generator(
    generator=batches.keras(train_paths, train_angles, augmentor=utils.augment_set, args=params.args),
    samples_per_epoch=800*params.batch_size,
    nb_epoch=params.max_epochs,
    validation_data=batches.validation(val_paths, val_angles, utils.flip_set),
    nb_val_samples=2*val_paths.shape[0],
    callbacks=callbacks
  )
