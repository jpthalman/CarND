import os
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import label

import VehicleDetection.classifier as classifier
from VehicleDetection.processing import single_img_features, get_windows, draw_boxes


class CarDetector(object):
    def __init__(self, model, scaler, im_size, frame_memory, threshold):
        self.model = model
        self.scaler = scaler
        self.im_size = im_size
        self.threshold = threshold
        self.windows = get_windows(y_start_stop=(400, 650))
        self.frame_memory = frame_memory
        self.frame_buffer = []

    def find_cars(self, im):
        heatmap = np.zeros_like(im[..., 0])
        features = []

        for w_coords in self.windows:
            roi = self._extract_window(im, w_coords)
            roi = cv2.resize(roi, self.im_size)
            features.append(single_img_features(roi))

        features = self.scaler.transform(features)

        for feat, coords in zip(features, self.windows):
            if self.model.predict(feat.reshape(1, -1))[0]:
                heatmap = self._add_heat(heatmap, coords)

        self._add_to_buffer(heatmap)
        avg_heatmap = self._get_heatmap_from_buffer()
        labels = label(avg_heatmap)
        return self._draw_labeled_bounding_boxes(im, labels)

    def _add_to_buffer(self, heat_map):
        if len(self.frame_buffer) == self.frame_memory:
            self.frame_buffer.pop(0)
        self.frame_buffer.append(heat_map)
        return

    def _get_heatmap_from_buffer(self):
        heatmap_sum = np.zeros_like(self.frame_buffer[0])
        for heatmap in self.frame_buffer:
            heatmap_sum = np.add(heatmap_sum, heatmap)
        heatmap_sum[heatmap_sum < self.threshold] = 0
        return heatmap_sum

    @staticmethod
    def _extract_window(im, window):
        xstart, ystart = window[0]
        xend, yend = window[1]
        return im[ystart:yend, xstart:xend, ...]

    @staticmethod
    def _add_heat(heatmap, window):
        xstart, ystart = window[0]
        xend, yend = window[1]
        heatmap[ystart:yend, xstart:xend, ...] += 1
        return heatmap

    @staticmethod
    def _draw_labeled_bounding_boxes(im, labels):
        for car_number in range(1, labels[1]+1):
            nonzero = (labels[0] == car_number).nonzero()
            nonzero_y = np.array(nonzero[0])
            nonzero_x = np.array(nonzero[1])

            bbox = ((np.min(nonzero_x), np.min(nonzero_y)), (np.max(nonzero_x), np.max(nonzero_y)))
            cv2.rectangle(im, bbox[0], bbox[1], (0, 0, 255), 6)
        return im


if __name__ == 'builtins':
    test_path = os.getcwd() + '/VehicleDetection/test_images/test6.jpg'

    if os.path.exists('VehicleDetection/model.p'):
        print('Loading the model from the pickled file...')
        with open('VehicleDetection/model.p', 'rb') as f:
            model, X_scaler = pickle.load(f)
        print('Done!')
    else:
        model, X_scaler = classifier.train()

    # im = cv2.imread(test_path)
    #
    # windows = get_windows(y_start_stop=(400, 650))
    # tmp = draw_boxes(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), windows, thick=2)
    # plt.imshow(tmp)

    im = cv2.imread(test_path)
    detector = CarDetector(model, X_scaler, (64, 64), 5, 3)
    for _ in range(5):
        out = detector.find_cars(im)

    cv2.imshow('', out)
