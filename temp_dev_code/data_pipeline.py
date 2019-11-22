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
    (image_paths, bbox, joints, scores)).shuffle(100)
root_path = os.path.abspath('.')

# first step
data = []
for sample in ds.take(1):
    for i in sample:
        data.append(i)
        print(i.numpy())

# second step
config = config()

img = plt.imread(dbcfg.img_path + os.sep + data[0].numpy().decode("utf-8"))
cropped_img, target_coord = image_process.cropped_image_and_pose_coord(
    data[0], data[1], data[2])

# plt.imshow(img)
# plt.show()
# plt.imshow(cropped_img)
# plt.show()

kps_array = np.array(data[2]).reshape((17, 3))
plt.imshow(dbcfg.vis_keypoints(img, kps_array.T))
plt.show()

imaga_path = dbcfg.img_path


def data_process(file_path, bbox, joints, scores):
    [cropped_img, target_coord] = tf.py_function(image_process.cropped_image_and_pose_coord,
                                                 [file_path, bbox, joints], [tf.float32, tf.int16])
    #heatmap = image_process.render_gaussian_heatmap(target_coord)
    return cropped_img, target_coord


processed_ds = ds.shuffle(100).map(data_process).batch(32)

for images, coords in processed_ds.take(1):
    print('mini_batch images : ', images.shape)
    print('mini_batch labels : ', coords.shape)
    print('coor type', coords)

plt.imshow(images[0].numpy().astype(int))
plt.show()

k = 0
vis = images[k].numpy().astype(int).copy()
for i in range(config.num_kps):
        #resized = cv2.resize(images[k].numpy().astype(int)/255., (images[k].shape[0]//4, images[k].shape[1]//4))
    resized = vis
    point = (coords[k][i][::-1].numpy()[0], coords[k][i][::-1].numpy()[1])
    print(point)
    cv2.circle(resized, point, 5, (255, 0, 0), -1)
    plt.imshow(resized)
    plt.show()
