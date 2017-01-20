import numpy as np
import cv2
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def load_data(file):
    """
    Opens the driving_log.csv and returns the center, left, right, and steering in a dictionary.

    :param file: Full file path to driving_log.csv
    :type file: String
    :return: Dictionary containing the camera file paths and steering angles.
    :rtype: Dictionary with keys = ['angles', 'center', 'left', 'right']
    """
    df = pd.read_csv(file, names=['CenterImage', 'LeftImage', 'RightImage', 'SteeringAngle',
                                  'Throttle', 'Break', 'Speed'])
    data = {
        'angles': df['SteeringAngle'].astype('float32').as_matrix(),
        'center': [im.replace(' ', '') for im in df['CenterImage'].as_matrix()],
        'right': [im.replace(' ', '') for im in df['RightImage'].as_matrix()],
        'left': [im.replace(' ', '') for im in df['LeftImage'].as_matrix()]
      }
    return data


def split_data(features, labels, test_size=0.2, shuffle_return=True):
    if shuffle_return:
        features, labels = shuffle(features, labels)

    return train_test_split(features, labels, test_size=test_size)


def augment_image(image, value, prob):
    """
    Augments an image and steering angle with probability `prob`.

    This technique randomly adjusts the brighness, occludes the image with 30 random black squares,
    and slightly rotates, shifts and scales the image. These augmentations are meant to make the
    model more robust to differing conditions than those in the training set.

    :param image: The image to augment
    :param value: The steering angle associated with the image
    :param prob: The probability of augmenting the image
    :return: Tuple with (augmented_image, augmented_value)

    :type image: np.ndarray
    :type value: float
    :type prob: float [0.0, 1.0]
    :rtype: tuple with values (augmented_image, augmented_value)
    """
    h, w, c = image.shape

    # Grayscale and normalize image to [-0.5, 0.5]
    if c == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image/255. - 0.5

    random_uniform = np.random.uniform(0.0, 1.0)

    # Return un-augmented image and value with probability (1-prob)
    if random_uniform > prob:
        return image, value

    # Flip the image and angle with probability (prob/2), aka 50% of the time.
    if random_uniform < prob/2:
        image, value = cv2.flip(image, 1), value
        image = image[..., None]

    # Random brightness adjustment
    image += np.random.uniform(-0.3, 0.3)

    # Random occlusion with dark squares
    sq_w, sq_h, sq_count = 25, 25, 30
    for i in range(sq_count):
        pt1 = (np.random.randint(0, w), np.random.randint(0, h))
        pt2 = (pt1[0] + sq_w, pt1[1] + sq_h)
        cv2.rectangle(image, pt1, pt2, (-0.5, -0.5, -0.5), -1)

    # Rotation/Scaling matrix
    rotation, scale = 1, 0.02
    M_rot = cv2.getRotationMatrix2D(
        (h//2, w//2),
        np.random.uniform(-rotation, rotation),
        np.random.uniform(1.0 - scale, 1.0 + scale)
      )

    # Shifts/Affine transforms
    src = np.array([[0,0], [w,0], [w,h]]).astype('float32')

    pixel_shift = 2
    x_shift = np.random.randint(-pixel_shift, pixel_shift)
    y_shift = np.random.randint(-pixel_shift, pixel_shift)

    dst = np.array([
        [0 + x_shift, 0 + y_shift],
        [w + x_shift, 0 + y_shift],
        [w + x_shift, h + y_shift]
      ]).astype('float32')

    M_affine = cv2.getAffineTransform(src, dst)

    # Apply augmentations to the image
    augmented = cv2.warpAffine(image, M_rot, (w,h))
    augmented = cv2.warpAffine(augmented, M_affine, (w,h))

    # Ensure there is a color channel
    if augmented.ndim < 3:
        augmented = augmented[..., None]

    # Add random noise to steering angle
    rand_ang = 0.005
    value += np.random.uniform(-rand_ang, rand_ang)
    return augmented, value


def augment_set(data, values, prob):
    """
    Applies `augment_image` to all images in a given batch.

    :param data: Batch of images
    :param values: Batch of corresponding values
    :return: Tuple containing (augmented_images, augmented_values)

    :type data: np.ndarray with shape (N, h, w, ch)
    :type values: np.ndarray with shape (N,)
    ":rtype: (np.ndarray, np.ndarray)
    """
    n_obs = data.shape[0]
    assert n_obs == values.shape[0], 'Different # of data and labels.'

    aug_data, aug_vals = [], []
    for i, img in enumerate(data):
        tmp = augment_image(img, values[i], prob)
        aug_data.append(tmp[0])
        aug_vals.append(tmp[1])

    return np.array(aug_data), np.array(aug_vals)


class BatchGenerator(object):
    def __init__(self, batch_size, augmentor=lambda x:x, args=()):
        self.batch_size = batch_size
        self.augmentor = augmentor
        self.args = args

    def __call__(self, data, values, load=False, path=''):
        n_obs = data.shape[0]
        assert n_obs == values.shape[0], 'Different # of data and labels.'

        for batch in range(0, n_obs, self.batch_size):
            batch_x = data[batch:min(batch + self.batch_size, n_obs), ...]
            if load:
                batch_x = np.array([cv2.imread(path + im) for im in batch_x])
            batch_y = values[batch:min(batch + self.batch_size, n_obs)]
            yield self.augmentor(batch_x, batch_y, *self.args)

    def keras(self, data, values, load=False, path=''):
        n_obs = data.shape[0]
        assert n_obs == values.shape[0], 'Different # of data and labels.'

        while True:
            for batch in range(0, n_obs, self.batch_size):
                batch_x = data[batch:min(batch + self.batch_size, n_obs), ...]
                if load:
                    batch_x = np.array([cv2.imread(path + im) for im in batch_x])
                batch_y = values[batch:min(batch + self.batch_size, n_obs)]
                yield self.augmentor(batch_x, batch_y, *self.args)
