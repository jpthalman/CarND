import numpy as np
import cv2
import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def split_data(features, labels, test_size=0.2, shuffle_return=True):
    if shuffle_return:
        features, labels = shuffle(features, labels)

    return train_test_split(features, labels, test_size=test_size)


def flip_and_concat(images, angles):
    n_obs = images.shape[0]
    assert n_obs == angles.shape[0], "Different number of images and steering angles"

    flipped_images = np.array([cv2.flip(image, 1)[..., None] for image in images])
    flipped_angles = np.array([-angle for angle in angles])

    images = np.concatenate((images, flipped_images), axis=0)
    angles = np.concatenate((angles, flipped_angles), axis=0)
    return shuffle(images, angles)


def preprocessed_image(path):
    image = cv2.imread(path)
    resized = cv2.resize(image, (64, 32))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return gray[..., np.newaxis]


def transform_images(images, angles):
    pass


def load_data(file):
    if not os.path.exists("/home/japata/sharefolder/CarND/Projects/P3-BehavioralCloning/data.p"):
        data = pd.read_csv(file,
                           names=['CenterImage', 'LeftImage', 'RightImage', 'SteeringAngle', 'Throttle', 'Break',
                                  'Speed'])
        center_paths = data['CenterImage'].as_matrix()
        right_paths = data['RightImage'].as_matrix()
        left_paths = data['LeftImage'].as_matrix()
        steering_angles = data['SteeringAngle'].astype('float32').as_matrix()

        path = "/home/japata/sharefolder/CarND/Projects/P3-BehavioralCloning/UdacityData/"
        c_images = np.array([preprocessed_image(path + im.replace(' ', '')) for im in center_paths])
        r_images = np.array([preprocessed_image(path + im.replace(' ', '')) for im in right_paths])
        l_images = np.array([preprocessed_image(path + im.replace(' ', '')) for im in left_paths])

        images = np.concatenate((c_images, r_images, l_images), axis=0)
        steering_angles = np.concatenate((steering_angles, steering_angles-0.25, steering_angles+0.25), axis=0)

        X_train, X_val, y_train, y_val = split_data(images, steering_angles, test_size=0.2, shuffle_return=True)

        X_train, y_train = flip_and_concat(X_train, y_train)
        X_val, y_val = flip_and_concat(X_val, y_val)

        with open('data.p', 'wb') as f:
            pickle.dump({'train':(X_train, y_train), 'val':(X_val, y_val)}, f)
    else:
        with open('data.p', 'rb') as f:
            data = pickle.load(f)
        X_train, y_train = data['train']
        X_val, y_val = data['val']

    return X_train, y_train, X_val, y_val

