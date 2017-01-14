import numpy as np


def load_data():
    import cv2
    import pandas as pd

    data = pd.read_csv("~/sharefolder/CarND/Projects/P3-BehavioralCloning/Data/driving_log.csv",
                       names=['CenterImage', 'LeftImage', 'RightImage', 'SteeringAngle', 'Throttle', 'Break', 'Speed'])
    image_paths, steering_angles = data['CenterImage'].as_matrix(), data['SteeringAngle'].as_matrix()

    images = []
    for path in image_paths:
        images.append(cv2.imread(path))

    return np.array(images), steering_angles

