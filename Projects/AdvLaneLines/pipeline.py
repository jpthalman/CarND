import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

from processing import calibrate_camera, undistort_img, colorspace_threshold, gradient_threshold, \
                       transform_perspective, sliding_window, get_return_values, predict_from_margin_around_prev_fit, \
                       get_curvature
from checks import roughly_parallel, similar_curvature


class LaneFinder(object):
    def __init__(self, M, dist, src, dst, debug=False):
        self.M = M
        self.dist = dist
        self.src = src
        self.dst = dst
        self.debug = debug
        self.left_prev = None
        self.right_prev = None

    def __call__(self, im):
        undistorted = undistort_img(im, self.M, self.dist)

        color_thresh = colorspace_threshold(undistorted, cv2.COLOR_RGB2HSV, 2, (225, 255), clahe=True)
        grad_thresh = gradient_threshold(undistorted, cv2.COLOR_RGB2HSV, 2, 'x', (10, 30), 3)

        combined_thresh = np.zeros_like(grad_thresh)
        combined_thresh[(grad_thresh == 1) | (color_thresh == 1)] = 1

        h, w = color_thresh.shape[:2]

        top_down = transform_perspective(combined_thresh, (w, h), self.src, self.dst)

        # Get the predicted lane lines and curvature using the previous fit
        left_fit, right_fit, l_curv, r_curv = predict_from_margin_around_prev_fit(
            top_down,
            self.left_prev,
            self.right_prev
          )

        # If the fit is not valid, use the sliding window technique
        if left_fit is None or not self.__valid_fit(left_fit, right_fit, l_curv, r_curv):
            left_fit, right_fit, l_curv, r_curv = sliding_window(top_down, 9)

        # Store fit for next prediction
        self.left_prev, self.right_prev = left_fit, right_fit

        # Get fitted points and plot
        y_axis = np.arange(ymax, dtype=np.float32)
        x_left, x_right = get_return_values(y_axis, left_fit), get_return_values(y_axis, right_fit)

        warp_zero = np.zeros_like(top_down).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([x_left, y_axis]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([x_right, y_axis])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space
        new_warp = transform_perspective(color_warp, (w, h), dst, src)

        # Combine the result with the original image
        if self.debug:
            color_thresh = np.dstack((np.zeros_like(grad_thresh), grad_thresh, color_thresh))*255
            return cv2.addWeighted(color_thresh, 1, new_warp, 0.3, 0)
        else:
            text = 'Curvature: %5.2f' % np.mean((l_curv, r_curv))
            output = cv2.putText(undistorted, text, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,)*3)
            return cv2.addWeighted(output, 1, new_warp, 0.3, 0)

    @staticmethod
    def __valid_fit(left, right, l_curv, r_curv):
        checks_passed = [
            roughly_parallel(left, right, 0.9)
          ]
        return all(checks_passed)


if __name__ == '__main__':
    ret, M, dist, rvecs, tvecs = calibrate_camera('\\camera_cal\\', 9, 6)

    test_im = os.getcwd() + '\\test_images\\test5.jpg'
    im = cv2.imread(test_im)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    ymax, xmax = im.shape[:2]
    x_shift = 80
    src = np.float32([
        [0, 720],
        [xmax // 2 - x_shift, 450],
        [xmax // 2 + x_shift, 450],
        [xmax, 720]
    ])

    shift = 100
    dst = np.float32([
        [shift, ymax],
        [shift, shift],
        [xmax - shift, shift],
        [xmax - shift, ymax]
    ])

    # detected = find_lane_lines(im, M, dist, src, dst)
    # plt.imshow(detected)

    lane_finder = LaneFinder(M, dist, src, dst, debug=False)
    project_video_output = 'project_video_output.mp4'

    try: os.remove(project_video_output)
    except FileNotFoundError: pass

    original = VideoFileClip('project_video.mp4')
    processed = original.fl_image(lane_finder)
    processed.write_videofile(project_video_output, audio=False)
