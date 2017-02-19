import utils
import cv2
from collections import namedtuple
import numpy as np
import os
import matplotlib.pyplot as plt


def test_generator(batch, shape):
    ims, angs = batch
    fig = plt.figure(figsize=(15, 15))

    for i, im in enumerate(ims):
        ax = fig.add_subplot(*shape, i+1)
        ax.set_title('%0.4f' % (0.25*angs[i]))
        ax.axis('off')
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_HSV2RGB), cmap='gray', vmin=0, vmax=255)
    fig.tight_layout()
    fig.show()


path = '/home/japata/sharefolder/CarND/Projects/BehavioralCloning/'

center_paths, center_angs = utils.concat_all_cameras(
    data=utils.load_data(path + 'Data/Center/', 'driving_log.csv'),
    angle_shift=0.2,
    condition_lambda=lambda x: abs(x) < 1e-5,
    keep_percent=0.1
  )

gen = utils.batch_generator(
    ims=center_paths,
    angs=center_angs,
    batch_size=64,
    augmentor=utils.augment_image,
    path='',
    kwargs={'prob': 1.0}
  )

# tmp = np.histogram(online_angs, np.arange(-1., 1., 0.01))[0]
# plt.bar(range(len(tmp)), tmp)

ims, angs = gen.__next__()

test_generator((ims, angs), (8,8))

im = 42
tmp = cv2.cvtColor(ims[im, ...], cv2.COLOR_HSV2RGB)
grid = plt.figure()
grid.suptitle(angs[im])
plt.imshow(tmp)

# -0.2318872
neg = '/home/japata/sharefolder/CarND/Projects/BehavioralCloning/Data/Center/IMG/center_2017_01_21_23_55_53_723.jpg'
# 0.2604084
pos = '/home/japata/sharefolder/CarND/Projects/BehavioralCloning/Data/Center/IMG/center_2017_01_21_23_57_37_575.jpg'

ang = -0.2319
im = cv2.imread(neg)
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

dark = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
dark = dark.astype(np.float32)
dark[..., 2] = add_random_shadow(dark[..., 2])
dark = dark.astype(np.uint8)
dark = cv2.cvtColor(dark, cv2.COLOR_HSV2RGB)


fig = plt.figure()
ax = fig.add_subplot(1,2,1)
ax.set_title('Original')
ax.axis('off')
ax.imshow(im)

ax = fig.add_subplot(1,2,2)
ax.set_title('Shadowed')
ax.axis('off')
ax.imshow(dark)

fig.tight_layout()
fig.show()
