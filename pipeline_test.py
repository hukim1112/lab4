import os
import os.path as osp
import sys
import numpy as np
import cv2
from data_util.COCO import dataset, image_process
import tensorflow as tf
import tensorflow_addons as tfa
from matplotlib import pyplot as plt
from config.coco_config import config

dbcfg = dataset.Dataset()
val_data = dbcfg.load_val_data_with_annot()

image_paths = [i['imgpath'] for i in val_data]
bbox = [i['bbox'] for i in val_data]
joints = [i['joints'] for i in val_data]
scores = [i['score'] for i in val_data]

ds = tf.data.Dataset.from_tensor_slices(
    (image_paths, bbox, joints, scores)).shuffle(1000)
root_path = os.path.abspath('.')

# first step
data = []
for sample in ds.take(1):
    for i in sample:
        data.append(i)
        # print(i.numpy())

# second step
config = config()
img = plt.imread(dbcfg.img_path + os.sep + data[0].numpy().decode("utf-8"))
bbox = data[1]
kps_sample = np.reshape(np.array(data[2]), (17, 3))
print(kps_sample)
for i, point in enumerate(kps_sample):
    cv2.circle(img, (point[0], point[1]), 5, (255, 0, 0), 3)
    cv2.putText(img, str(i), (point[0], point[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
plt.imshow(img)
plt.show()

cropped_img, target_coord = image_process.cropped_image_and_pose_coord(
    data[0], data[1], data[2])

print(target_coord)
vis = cropped_img.copy()
for i, point in enumerate(target_coord):
    cv2.circle(vis, (point[0], point[1]), 2, (255, 0, 0), -1)
    cv2.putText(vis, str(i), (point[0], point[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

heatmap = image_process.render_gaussian_heatmap(
    target_coord, cropped_img.shape[:2])
print(heatmap.shape)

for i in range(config.num_kps):
    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.imshow(vis)
    fig.add_subplot(1, 2, 2)
    plt.imshow(heatmap[:, :, i])
    plt.show()
