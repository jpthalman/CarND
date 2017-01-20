"""
Dependencies:
    Numpy
    OpenCV 3
    Pandas
    SciKit-Learn
    Keras
"""
from collections import namedtuple

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Lambda
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras.layers.advanced_activations import ELU
from keras.optimizers import adam

# Load the data
path = '/home/japata/sharefolder/CarND/Projects/P3-BehavioralCloning/UdacityData/'


# Model construction
model = Sequential([
    Convolution2D(3, 1, 1, border_mode='valid', init='he_normal'),

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
print('\n', model.summary(), '\n')
optimizer = adam(
    lr=params.learning_rate,
    epsilon=params.epsilon,
    decay=params.decay
  )
model.compile(optimizer=optimizer, loss='mse')


# Model training

model.fit(X_train, y_train, batch_size=256, nb_epoch=10, validation_data=(X_val, y_val))


# Save model weights and structure
with open('model.json', 'w') as file:
    file.write(model.to_json())
model.save_weights('model.h5')

