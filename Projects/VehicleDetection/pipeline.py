import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import VehicleDetection.classifier as classifier
from VehicleDetection.processing import single_img_features, slide_window


class CarDetector(object):
    def __init__(self, model, frame_memory):
        self.model = model
        self.frame_memory = frame_memory
        self.frame_buffer = []

    def add_to_buffer(self, heat_map):
        if len(self.frame_buffer) == self.frame_memory:
            self.frame_buffer.pop(0)
        self.frame_buffer.append(heat_map)
        return


if __name__ == 'builtins':
    test_path = os.getcwd() + '\\VehicleDetection\\test_images\\test1.jpg'

    im = cv2.imread(test_path)
    plt.imshow(im[..., [2,1,0]])
    im = cv2.cvtColor(im, cv2.COLOR_BGR2LUV)
    im = im[400:510, 1050:, :]
    im = cv2.resize(im, (64, 64))
    features = single_img_features(im)

    model = classifier.train()
    print(model.predict(np.expand_dims(features, 0)))
