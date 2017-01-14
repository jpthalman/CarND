import numpy as np
import cv2


def load_data(file):
    import pandas as pd

    data = pd.read_csv(file, names=['CenterImage', 'LeftImage', 'RightImage', 'SteeringAngle', 'Throttle', 'Break',
                                    'Speed'])
    image_paths, steering_angles = data['CenterImage'].as_matrix(), data['SteeringAngle'].as_matrix()

    images = []
    for path in image_paths:
        images.append(cv2.imread(path))

    return np.array(images), steering_angles


def split_data(features, labels, test_size=0.2, shuffle=True):
    from sklearn.model_selection import train_test_split
    from sklearn.utils import shuffle

    if shuffle:
        features, labels = shuffle(features, labels)

    return train_test_split(features, labels, test_size=test_size)


def flip_and_concat(images, angles):
    n_obs = images.shape[0]
    assert n_obs == angles.shape[0], "Different number of images and steering angles"

    flipped_images = []
    flipped_angles = []
    for i, image in enumerate(images):
        flipped_images.append(cv2.flip(image, 1))
        flipped_angles.append(-angles[i])

    images = np.concatenate((images, np.array(flipped_images)), axis=0)
    angles = np.concatenate((angles, np.array(flipped_angles)), axis=0)
    return images, angles


def transform_images(images, angles):
    pass

