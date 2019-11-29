import os
import os.path as osp
import sys
import numpy as np
import cv2
import tensorflow as tf
from matplotlib import pyplot as plt

from data_util.COCO import dataset, image_process
from config.coco_config import config


#COCO dataset load
dbcfg = dataset.Dataset()
val_data = dbcfg.load_val_data_with_annot()

image_paths = [i['imgpath'] for i in val_data]
bbox = [i['bbox'] for i in val_data]
joints = [i['joints'] for i in val_data]

ds = tf.data.Dataset.from_tensor_slices(
    (image_paths, bbox, joints)).shuffle(1000)

#Sample a example from dataset object
kps_available = True
while kps_available == True:
    sampled_data = next(iter(ds))
    if sampled_data[2].numpy().sum() != 0:
        kps_available = False

#Visualization of keypoints on original image
config = config()
img_path = osp.join(config.image_path, sampled_data[0].numpy().decode("utf-8"))
img = plt.imread(img_path)
bbox = sampled_data[1]
kps = np.reshape(np.array(sampled_data[2]), (17, 3))

circle_size = int(img.shape[0]/100)
font_size = img.shape[0]/1000
font_width = int(img.shape[0]/400)
for i, point in enumerate(kps):
    cv2.circle(img, (point[0], point[1]), circle_size, (255, 0, 0), -1)
    cv2.putText(img, str(i), (point[0], point[1]),
                cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), 1)

#Test rendering of heatmap of the example
cropped_img, target_coord = image_process.cropped_image_and_pose_coord(
    sampled_data[0], sampled_data[1], sampled_data[2])

circle_size = int(cropped_img.shape[0]/100)
font_size = cropped_img.shape[0]/1000
font_width = int(cropped_img.shape[0]/400)
vis = cropped_img.copy()
for i, point in enumerate(target_coord):
    cv2.circle(vis, (point[0], point[1]), circle_size, (255, 0, 0), -1)
    cv2.putText(vis, str(i), (point[0], point[1]),
                cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), 1)


fig = plt.figure()
fig.add_subplot(1, 2, 1)
plt.imshow(img)
plt.title("Plot keypoints on original image")
fig.add_subplot(1, 2, 2)
plt.imshow(vis)
plt.title("Plot keypoints on cropped image")
plt.show()


heatmaps = image_process.render_gaussian_heatmap(target_coord, cropped_img.shape[:2])

fig = plt.figure()
spec = fig.add_gridspec(ncols = 5+4, nrows=4)
fig.add_subplot(spec[:4, :4])
plt.imshow(vis)

for i in range(config.num_kps):
    row = i//5
    col = i%5 + 4
    #print(row, col, type(row), type(col))
    fig.add_subplot(spec[row, col])
    plt.imshow(heatmaps[:, :, i])
plt.show()
