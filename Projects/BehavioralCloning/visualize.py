import os
import sys
import argparse
import numpy as np
import scipy
import cv2
import pandas as pd
from moviepy.editor import ImageSequenceClip

import keras.backend as K
from keras.models import model_from_json


class VisualizeActivations(object):
    def __init__(self, model, preprocessor, epsilon=1e-7):
        self.model = model
        self.preprocessor = preprocessor
        self.epsilon = epsilon
        self.layer_dict = dict([(layer.name, layer) for layer in self.model.layers])

    def heat_map(self, layer_name, img_path):
        im = cv2.imread(img_path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        w, h = im.shape[:2]

        processed = self.preprocessor(im)

        pred_angle = K.sum(self.model.layers[-1].output)
        layer = self.layer_dict[layer_name]
        grads = K.gradients(pred_angle, layer.output)[0]

        gradients_function = K.function([self.model.layers[0].input], [self.model.output, grads, pred_angle])

        conv_outputs, grads_val, angle = gradients_function([processed])
        conv_outputs, grads_val = conv_outputs[0, :], grads_val[0, ...]

        class_weights = self.grad_cam_loss(grads_val, angle)

        # Create the class activation map
        cam = np.mean(class_weights*conv_outputs, axis=2)
        cam /= np.max(cam)
        cam = cv2.resize(cam, (h, w))

        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap[np.where(cam < 0.2)] = 0

        output = heatmap * 0.5 + im

        length = 50
        color = (0, 255, 0)
        xshift = length*np.cos(angle)*np.sign(angle)
        yshift = length*np.sin(angle)

        output = cv2.line(output, (w//2, h), (w//2 + xshift, h-yshift), color)
        return output

    def grad_cam_loss(self, x, angle):
        if angle > 5.0 * scipy.pi / 180.0:
            return x
        elif angle < -5.0 * scipy.pi / 180.0:
            return -x
        else:
            x += self.epsilon
            return (1.0 / x) * np.sign(angle)


def load_data(path, file):
    """
    Opens driving_log.csv and returns center, left, right, and steering in a dictionary.

    :param path: Full file path to file
    :param file: The name of the file to load

    :type path: String
    :type file: String

    :return: Dictionary containing the camera file paths and steering angles.
    :rtype: Dictionary with keys = ['angles', 'center', 'left', 'right']
    """
    df = pd.read_csv(path + file, names=['CenterImage', 'LeftImage', 'RightImage', 'SteeringAngle',
                                         'Throttle', 'Break', 'Speed'])
    data = {
        'angles': df['SteeringAngle'].astype('float32').as_matrix(),
        'center': np.array([path + str(im).replace(' ', '').replace('\\', '/') for im in df['CenterImage'].as_matrix()]),
        'right': np.array([path + str(im).replace(' ', '').replace('\\', '/') for im in df['RightImage'].as_matrix()]),
        'left': np.array([path + str(im).replace(' ', '').replace('\\', '/') for im in df['LeftImage'].as_matrix()])
      }
    return data


def process_image(im):
    """
    Crop image, convert to HSV, and resize.

    :param im: Image to normalize
    :return: Normalized image with shape (h, w, ch)

    :type im: np.ndarray with shape (h, w, 3)
    :rtype: np.ndarray with shape (h, w, ch)
    """
    assert im.ndim == 3 and im.shape[2] == 3, 'Must be a BGR image with shape (h, w, 3)'

    im = im[50:135, :]
    im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    im = cv2.resize(im, (64, 64))

    if im.ndim == 2:
        im = np.expand_dims(im, -1)
    return im


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', help='The filepath to the JSON file for the model.')
    parser.add_argument('--h5', help='The filepath to the H5 file for the model')
    parser.add_argument('--log', help='The filepath to driving_log.csv')
    parser.add_argument('--layer', help='Name of the Conv Layer of which to visualize the activations')
    parser.add_argument('--fps', default=30, help='FPS for output video')
    parser.add_argument('--dir', default=os.getcwd() + '\\',
                        help='Optional filepath to set the current working directory')
    parser.add_argument('--layer-names', action='store_true',
                        help='Flag to print out the layer names of the model and stop execution')

    args = parser.parse_args()

    # Load the model from the args.
    with open(args.dir + args.json, 'r') as jfile:
        model = model_from_json(jfile.read())
    model.compile("adam", "mse")
    model.load_weights(args.h5)

    # Print layer names and exit, if requested.
    if args.layer_names:
        model.summary()
        sys.exit()

    activation = VisualizeActivations(model=model, preprocessor=lambda x: x)
    data = load_data(args.dir, args.log)

    frames = [activation.heat_map(args.layer, im_path) for im_path in data['center']]
    video = ImageSequenceClip(frames, fps=args.fps)
    video.write_videofile('activation_heatmap.mp4')
