import os
import pdb
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip

import VehicleDetection.classifier as classifier
from VehicleDetection.processing import bin_spatial, get_hog_features, color_hist


class CarDetector(object):
    def __init__(self,
                 model,
                 scaler,
                 im_size,
                 ystart,
                 ystop,
                 scale,
                 orient,
                 pix_per_cell,
                 cell_per_block,
                 spatial_size,
                 hist_bins,
                 frame_memory,
                 threshold):
        """
        This class is meant to take in an image, process it in a predefined way, and return the same image with
        a bounding box around any cars in the image.

        :param model: Pre-trained model to identify cars in a ROI.
        :param scaler: Scaler used to normalize image data when training the model.
        :param im_size: Size of image to feed the model.
        :param ystart: Upper bound pixel on Y-axis to search for cars.
        :param ystop: Lower bound pixel on Y-axis to search for cars.
        :param scale: Factor by which to scale the search ROI. Should be 1.
        :param orient: Number of gradient bins for HOG features
        :param pix_per_cell: For HOG features
        :param cell_per_block: For HOG features
        :param spatial_size: Size transform tuple for the spatial features
        :param hist_bins: Number of histogram bins for color_hist
        :param frame_memory: Number of frames in the past to `remember`. Helps reduce false positives.
        :param threshold: Remove all values less than this threshold from the heatmap. Helps reduce false positives.
        """
        self.model = model
        self.scaler = scaler
        self.im_size = im_size
        self.ystart = ystart
        self.ystop = ystop
        self.scale = scale
        self.orient = orient
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block
        self.spatial_size = spatial_size
        self.hist_bins = hist_bins
        self.threshold = threshold
        self.frame_memory = frame_memory
        self.frame_buffer = []

    def find_cars(self,
                  im):
        """
        Searches through the Y-range defined in the `init` using 64x64 blocks and steping 16 pixels at a time. Each
        block is fed through the model, and if the model identifies a car in the block, heat is added to a heatmap in
        that region. Once the heatmap has been constructed, it is added to the buffer, and averaged heatmap is
        constructed, and class labels are assigned to the maximums of this heatmap. Using the class labeled mask,
        bounding boxes are drawn around each class on the original image and this annotated image is returned.

        :param im: Original image.
        :return: Annotated image.
        """
        img = im.astype(np.float32) / 255
        heatmap = np.zeros_like(img)

        img_tosearch = img[self.ystart:self.ystop, ...]
        ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)
        if self.scale != 1:
            h, w, ch = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (w // self.scale, h // self.scale))

        ch1 = ctrans_tosearch[:, :, 0]
        ch2 = ctrans_tosearch[:, :, 1]
        ch3 = ctrans_tosearch[:, :, 2]

        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // self.pix_per_cell) - 1
        nyblocks = (ch1.shape[0] // self.pix_per_cell) - 1
        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // self.pix_per_cell) - 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step

        # Compute individual channel HOG features for the entire image
        hog1 = get_hog_features(ch1, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)
        hog2 = get_hog_features(ch2, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)

        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb * cells_per_step
                xpos = xb * cells_per_step
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                xleft = xpos * self.pix_per_cell
                ytop = ypos * self.pix_per_cell

                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], self.im_size)

                # Get color features
                spatial_features = bin_spatial(subimg, size=self.spatial_size)
                hist_features = color_hist(subimg, nbins=self.hist_bins)

                # Scale features and make a prediction
                test_features = self.scaler.transform(np.hstack((
                    spatial_features,
                    hist_features,
                    hog_features))).reshape(1, -1)
                # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
                test_prediction = self.model.predict(test_features)

                if test_prediction == 1:
                    xbox_left = np.int(xleft * self.scale)
                    ytop_draw = np.int(ytop * self.scale)
                    win_draw = np.int(window * self.scale)
                    top_left = (xbox_left, ytop_draw + self.ystart)
                    bottom_right = (xbox_left + win_draw, ytop_draw + win_draw + self.ystart)
                    self._add_heat(heatmap, (top_left, bottom_right))

        self._add_to_buffer(heatmap)
        avg_heatmap = self._get_heatmap_from_buffer()
        labels = label(avg_heatmap)
        return self._draw_labeled_bounding_boxes(im, labels)

    def _add_to_buffer(self, heat_map):
        """
        Adds a given heatmap to the heatmap buffer. If the buffer is currently the maximum length, remove the
        oldest element and then add the given heatmap.
        """
        if len(self.frame_buffer) == self.frame_memory:
            self.frame_buffer.pop(0)
        self.frame_buffer.append(heat_map)
        return

    def _get_heatmap_from_buffer(self):
        """
        Gets an averaged heatmap from every heatmap currently in the buffer.
        """
        heatmap_sum = np.zeros_like(self.frame_buffer[0])
        for heatmap in self.frame_buffer:
            heatmap_sum = np.add(heatmap_sum, heatmap)
        heatmap_sum[heatmap_sum < self.threshold] = 0
        return heatmap_sum

    @staticmethod
    def _add_heat(heatmap, window):
        """
        Adds `heat`, A.K.A. 1, to every pixel within a given window in a heatmap.

        :param heatmap: Heatmap to add heat to
        :param window: List containing the upper left and bottom right corners of the ROI to add heat to.
        """
        xstart, ystart = window[0]
        xend, yend = window[1]
        heatmap[ystart:yend, xstart:xend, ...] += 1
        return heatmap

    @staticmethod
    def _draw_labeled_bounding_boxes(im, labels):
        """
        Given a class labeled image mask, find the tightest bounding box around each class, draw these boxes onto the
        original image, and return the annotated image.

        :param im: Image to draw on
        :param labels: Pixel-wise class labeled mask of the original image.
        """
        cpy = np.copy(im)
        for car_number in range(1, labels[1]+1):
            nonzero = (labels[0] == car_number).nonzero()
            nonzero_y = np.array(nonzero[0])
            nonzero_x = np.array(nonzero[1])

            bbox = ((np.min(nonzero_x), np.min(nonzero_y)), (np.max(nonzero_x), np.max(nonzero_y)))
            cv2.rectangle(cpy, bbox[0], bbox[1], (0, 0, 255), 6)
        return cpy


if __name__ == '__main__':
    # Load the model from the pickle file if it exists, otherwise train it.
    if os.path.exists('VehicleDetection/model.p'):
        print('Loading the model from the pickled file...')

        with open('VehicleDetection/model.p', 'rb') as f:
            model, X_scaler = pickle.load(f)

        print('Done!')
    else:
        model, X_scaler = classifier.train()

    # Instantiate the detector
    detector = CarDetector(
        model=model,
        scaler=X_scaler,
        im_size=(64, 64),
        ystart=400,
        ystop=650,
        scale=1,
        orient=9,
        pix_per_cell=8,
        cell_per_block=2,
        spatial_size=(32, 32),
        hist_bins=32,
        frame_memory=5,
        threshold=1)

    infile = 'VehicleDetection/project_video.mp4'
    outfile = 'VehicleDetection/project_video_output.mp4'

    # Load and process the video
    original = VideoFileClip(infile)
    processed = original.fl_image(detector.find_cars)
    processed.write_videofile(outfile, audio=False)
