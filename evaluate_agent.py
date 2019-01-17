import numpy as np
from chainercv.visualizations import vis_bbox
from chainercv.utils import read_image
import matplotlib.pyplot as plt

img = read_image('../dataset-generator/images/image_7.png')

bboxes_all_images = np.load('../dataset-generator/bounding_boxes.npy')
bboxes_single_image = bboxes_all_images[7]
bboxes_chainercv = list()
for bbox in bboxes_single_image:
    x_min = bbox[0][0]
    y_min = bbox[0][1]
    x_max = bbox[1][0]
    y_max = bbox[1][1]
    bboxes_chainercv.append([y_min, x_min, y_max, x_max])
bboxes_chainercv = np.array(bboxes_chainercv, dtype=np.float32)

vis_bbox(img, bboxes_chainercv)
plt.show()
