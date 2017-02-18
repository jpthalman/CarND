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


K.set_learning_phase(0)


class VisualizeActivations(object):
    def __init__(self, model, preprocessor, rectifier, epsilon=1e-7):
        self.model = model
        self.preprocessor = preprocessor
        self.rectifier = rectifier
        self.epsilon = epsilon

        self.layer_dict = dict([(layer.name, layer) for layer in self.model.layers])
        self.gradients_function = None

    def set_gradient_function(self, layer_name):
        pred_angle = K.sum(self.model.layers[-1].output)
        layer = self.layer_dict[layer_name]
        grads = K.gradients(pred_angle, layer.output)[0]

        self.gradients_function = K.function(
            [self.model.layers[0].input],
            [self.model.output, grads, pred_angle]
          )
        return

    def heat_map(self, layer_name, img_path):
        if os.path.exists(img_path):
            im = cv2.imread(img_path)
        else:
            raise ValueError('Image does not exist:\n%r' % img_path)

        # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        processed = self.preprocessor(im)
        processed = np.expand_dims(processed, 0)

        w, h = im.shape[:2]

        if self.gradients_function is None:
            self.set_gradient_function(layer_name)

        conv_outputs, grads_val, angle = self.gradients_function([processed])
        conv_outputs, grads_val = conv_outputs[0, :], grads_val[0, ...]

        class_weights = self.grad_cam_loss(grads_val, angle)

        # Create the class activation map
        cam = np.mean(class_weights*conv_outputs, axis=2)
        cam /= np.max(cam)
        cam = self.rectifier(cam)

        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap[np.where(cam < 0.2)] = 0

        rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        output = cv2.addWeighted(rgb, 1, heatmap, 0.4, 0)

        length = 50
        color = (0, 255, 0)
        xshift = int(length*np.cos(np.deg2rad(90 + 25*angle)))
        yshift = int(length*np.sin(np.deg2rad(90 + 25*angle)))
        output = cv2.line(output, (h//2, w), (h//2 - xshift, w - yshift), color, thickness=2)
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


def processor(im):
    im = im[50:135, :]
    im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    return cv2.resize(im, (64, 64))


def rectifier(im):
    resized = cv2.resize(im, (320, 85))
    top_padding = np.zeros([50, 320])
    bot_padding = np.zeros([25, 320])
    output = np.concatenate((top_padding, resized, bot_padding))
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', help='The filepath to the JSON file for the model.')
    parser.add_argument('--h5', help='The filepath to the H5 file for the model')
    parser.add_argument('--log', help='The filepath to driving_log.csv')
    parser.add_argument('--layer', help='Name of the Conv Layer of which to visualize the activations')
    parser.add_argument('--fps', default=15, help='FPS for output video')
    parser.add_argument('--dir', default=os.getcwd() + '/',
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

    activation = VisualizeActivations(model=model, preprocessor=processor, rectifier=rectifier)
    data = load_data(args.dir + 'Data/Center/', args.log)

    frames = [activation.heat_map(args.layer, im_path) for im_path in data['center'][:1000]]
    video = ImageSequenceClip(frames, fps=args.fps)
    video.write_videofile('activation_heatmap.mp4')
