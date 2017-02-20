import os
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import VehicleDetection.classifier as classifier
from VehicleDetection.processing import single_img_features, get_windows


class CarDetector(object):
    def __init__(self, model, im_size, frame_memory, threshold):
        self.model = model
        self.im_size = im_size
        self.threshold = threshold
        self.windows = get_windows()
        self.frame_memory = frame_memory
        self.frame_buffer = []

        self.debug = None

    def find_cars(self, im):
        heatmap = np.zeros_like(im[..., 0])
        features = []

        for w_coords in self.windows:
            roi = self._extract_window(im, w_coords)
            roi = cv2.resize(roi, self.im_size)
            features.append(single_img_features(roi))

        X_scaler = StandardScaler().fit(features)
        features = X_scaler.transform(features)
        self.debug = features

        for feat, coords in zip(features, self.windows):
            if self.model.predict(feat.reshape(1, -1))[0]:
                heatmap = self._add_heat(heatmap, coords)

        self._add_to_buffer(heatmap)
        return self.get_heatmap_from_buffer()

    def _add_to_buffer(self, heat_map):
        if len(self.frame_buffer) == self.frame_memory:
            self.frame_buffer.pop(0)
        self.frame_buffer.append(heat_map)
        return

    def get_heatmap_from_buffer(self):
        heatmap_sum = np.zeros_like(self.frame_buffer[0])
        for heatmap in self.frame_buffer:
            heatmap_sum = np.add(heatmap_sum, heatmap)
        heatmap_sum[heatmap_sum < self.threshold] = 0
        return heatmap_sum

    @staticmethod
    def _extract_window(im, window):
        xstart, xend = window[0]
        ystart, yend = window[1]
        return im[ystart:yend, xstart:xend, ...]

    @staticmethod
    def _add_heat(heatmap, window):
        xstart, xend = window[0]
        ystart, yend = window[1]
        heatmap[ystart:yend, xstart:xend, ...] += 1
        return heatmap


if __name__ == '__main__':
    test_path = os.getcwd() + '/VehicleDetection/test_images/test1.jpg'

    if os.path.exists('VehicleDetection/model.p'):
        with open('VehicleDetection/model.p', 'rb') as f:
            model = pickle.load(f)
    else:
        model = classifier.train()

    im = cv2.imread(test_path)
    im = im[620:720, 0:100, :]
    im = cv2.resize(im, (64, 64))
    features = single_img_features(im)
    features = features.reshape(1, -1)

    X_scaler = StandardScaler().fit(features)
    features = X_scaler.transform(features)
    print(model.predict(features))

    im = cv2.imread(test_path)
    detector = CarDetector(model, (64, 64), 5, 75)
    for _ in range(5):
        heatmap = detector.find_cars(im)
    plt.imshow(heatmap, cmap='gray')
