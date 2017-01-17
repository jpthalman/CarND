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
from keras.layers.core import Dense, Dropout, Flatten, Lambda
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras.layers.advanced_activations import ELU, PReLU

# Load the data
path = "/home/japata/sharefolder/CarND/Projects/P3-BehavioralCloning/UdacityData/driving_log.csv"
print("Loading Data... ", end='')
X_train, y_train, X_val, y_val = utils.load_data(path)
print('Done!')

print('Observation Counts:\nTraining: %d | Validation: %d\n' % (X_train.shape[0], X_val.shape[0]))

# Model construction
model = Sequential()

model.add(Lambda(lambda x: x/255. - 0.5, input_shape=[32, 64, 1]))

model.add(Convolution2D(3, 1, 1, border_mode='valid', init='he_normal'))

model.add(Convolution2D(32, 5, 5, border_mode='valid', init='he_normal'))
model.add(MaxPooling2D((2,2), (2,2), 'valid'))
model.add(ELU())
model.add(Convolution2D(64, 5, 5, border_mode='valid', init='he_normal'))
model.add(MaxPooling2D((2,2), (2,2), 'valid'))
model.add(ELU())
model.add(Convolution2D(128, 5, 5, border_mode='valid', init='he_normal'))
model.add(ELU())

model.add(Flatten())
model.add(Dense(1024, init='he_normal'))
model.add(Dropout(0.5))
model.add(ELU())
model.add(Dense(256, init='he_normal'))
model.add(Dropout(0.5))
model.add(ELU())
model.add(Dense(1, init='he_normal'))

print('\n', model.summary(), '\n')

# Model training
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, batch_size=256, nb_epoch=10, validation_data=(X_val, y_val))


# Save model weights and structure
with open('model.json', 'w') as file:
    file.write(model.to_json())
model.save_weights('model.h5')

