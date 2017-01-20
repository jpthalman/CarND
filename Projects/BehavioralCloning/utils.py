import numpy as np
import cv2
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from decorators import n_images


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


def keep_n_percent_of_data_where(data, values, condition_lambda, percent):
    """
    Keeps n-percent of a dataset-values pair where a condition over the values is true.

    :param data: The dataset
    :param values: The values to filter over
    :param condition_lambda: A lambda by which to filter the dataset
    :param percent: The percent of the dataset-value pair (where `condition_lambda` is true) to KEEP. [0.0, 1.0].

    :type data: np.ndarray
    :type values: np.ndarray
    :type condition_lambda: lambda
    :type percent: float

    :return: Filtered tuple: (filtered_data, filtered_values)
    :rtype: Tuple
    """
    assert data.shape[0] == values.shape[0], 'Different # of data points and values.'

    cond_true = condition_lambda(values)
    data_true, data_false = data[cond_true, ...], data[~cond_true, ...]
    val_true, val_false = values[cond_true], values[~cond_true]

    cutoff = int(percent * data_true.shape[0])
    # Shuffle before clipping the top (1-n)%
    data_true, val_true = shuffle(data_true, val_true)
    # Only keep n% of the data points where the condition is true
    clipped_data_true, clipped_val_true = data_true[:cutoff, ...], val_true[:cutoff]

    filtered_data = np.concatenate((data_false, clipped_data_true), axis=0)
    filtered_values = np.concatenate((val_false, clipped_val_true), axis=0)

    return filtered_data, filtered_values


def split_data(features, labels, test_size=0.2, shuffle_return=True):
    if shuffle_return:
        features, labels = shuffle(features, labels)

    return train_test_split(features, labels, test_size=test_size)


def normalize_image(im):
    """
    Grayscale the image and normalize its pixels to [-0.5, 0.5]

    :param im: Image to normalize
    :return: Normalized image with shape (h, w, 1)
    """
    assert im.ndim == 3 and im.shape[2] == 3, 'Must be a BGR image with shape (h, w, 3)'

    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = im/255. - 0.5

    if im.ndim == 2:
        im = im[..., None]
    return im


@n_images
def flip_image(image, angle):
    """
    Mirrors the image from left to right and flips the sign of the angle.

    :param image: Image to flip
    :param angle: Angle to flip
    :return: (flipped_images, flipped_angle)
    """
    flipped = cv2.flip(image, 1)
    if flipped.ndim == 2:
        flipped = flipped[..., None]
    return flipped, -angle


def val_augmentor(ims, vals):
    """
    Normalizes images/vals into first set, flips and concats into second set, concats both sets, and returns.

    :param ims: Images to normalize/flip
    :param vals: Angles to normalize/flip
    :return: (normalized/flipped images, normalized/flipped angles)
    """
    normalized = np.array([normalize_image(im) for im in ims])
    flipped = flip_image(normalized)
    return np.concatenate((normalized, flipped), axis=0), \
           np.concatenate((vals, -vals), axis=0)


@n_images
def augment_image(image, value, prob, im_normalizer=normalize_image):
    """
    Augments an image and steering angle with probability `prob`.

    This technique randomly adjusts the brightness, occludes the image with 30 random black squares,
    and slightly rotates, shifts and scales the image. These augmentations are meant to make the
    model more robust to differing conditions than those in the training set.

    :param image: The image to augment
    :param value: The steering angle associated with the image
    :param prob: The probability of augmenting the image
    :param im_normalizer: Function to normalize the image
    :return: Tuple with (augmented_image, augmented_value)

    :type image: np.ndarray
    :type value: float
    :type prob: float [0.0, 1.0]
    :type im_normalizer: function
    :rtype: tuple with values (augmented_image, augmented_value)
    """
    assert image.ndim == 3, 'Image must have dimensions (h, w, ch)'

    image = im_normalizer(image)

    h, w = image.shape[:2]

    random_uniform = np.random.uniform(0.0, 1.0)

    # Flip the image and angle with probability (prob/2), aka 50% of the time.
    if random_uniform < prob/2:
        image, value = cv2.flip(image, 1), -value
        image = image[..., None]

    # Return un-augmented image and value with probability (1-prob)
    if random_uniform > prob:
        return image, value

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


# TODO: Delete if `augment_image` with decorator works
def augment_set(data, values, prob):
    """
    Applies `augment_image` to all images in a given batch.

    :param data: Batch of images
    :param values: Batch of corresponding values
    :param prob: The probability of augmenting an image in the batch
    :return: Tuple containing (augmented_images, augmented_values)

    :type data: np.ndarray with shape (N, h, w, ch)
    :type values: np.ndarray with shape (N,)
    :type prob: float between [0.0, 1.0]
    ":rtype: (np.ndarray, np.ndarray)
    """
    n_obs = data.shape[0]
    assert n_obs == values.shape[0], 'Different # of data and labels.'

    aug_batch_data, aug_batch_vals = [], []
    for i, img in enumerate(data):
        aug_data, aug_val = augment_image(img, values[i], prob)
        aug_batch_data.append(aug_data)
        aug_batch_vals.append(aug_val)
    return np.array(aug_batch_data), np.array(aug_batch_vals)


def batch_generator(ims, vals, batch_size, augmentor, path, args={}):
    n_obs = ims.shape[0]
    assert n_obs == vals.shape[0], 'Different # of data and labels.'

    while True:
        for batch in range(0, n_obs, batch_size):
            next_idx = batch + batch_size
            batch_x = ims[batch:min(next_idx, n_obs), ...]
            batch_y = vals[batch:min(next_idx, n_obs), ...]

            # Ensure consistent batch size by adding random images to the last
            # batch iff n_obs%batch_size != 0.
            if next_idx > n_obs:
                rand_idx = np.random.randint(0, n_obs-1, next_idx - n_obs)
                batch_x += ims[rand_idx, ...]
                batch_y += vals[rand_idx, ...]

            # Load the images from their paths
            batch_x = np.array([cv2.imread(path + im) for im in batch_x])
            # Augment the images with the given function
            batch_x, batch_y = augmentor(batch_x, batch_y, **args)
            yield batch_x, batch_y


# TODO: Delete if `batch_generator` works
class BatchGenerator(object):
    def __init__(self, batch_size, load=False, path=''):
        self.batch_size = batch_size
        self.load = load
        self.path = path

    def validation(self, data, values, augmentor=None):
        n_obs = data.shape[0]
        assert n_obs == values.shape[0], 'Different # of data and labels.'

        while True:
            for batch in range(0, n_obs, self.batch_size):
                batch_x = data[batch:min(batch + self.batch_size, n_obs), ...]
                batch_y = values[batch:min(batch + self.batch_size, n_obs)]

                if self.load:
                    batch_x = np.array([cv2.imread(self.path + im) for im in batch_x])

                yield batch_x, batch_y

                if augmentor is not None:
                    batch_x, batch_y = augmentor(np.array(batch_x), np.array(batch_y))
                    yield batch_x, batch_y

    def keras(self, data, values, augmentor=None, args={}):
        n_obs = data.shape[0]
        assert n_obs == values.shape[0], 'Different # of data and labels.'

        while True:
            batch_x, batch_y = [], []
            for _ in range(self.batch_size):
                idx = np.random.randint(0, n_obs-1)
                batch_x.append(data[idx, ...])
                batch_y.append(values[idx])

            batch_x, batch_y = np.array(batch_x), np.array(batch_y)

            if self.load:
                batch_x = np.array([cv2.imread(self.path + im) for im in batch_x])
            if augmentor is not None:
                batch_x, batch_y = augmentor(np.array(batch_x), np.array(batch_y), **args)
            yield batch_x, batch_y
