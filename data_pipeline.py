import os
import os.path as osp
import sys
import numpy as np
import cv2
from data_util.COCO import dataset, image_process
import tensorflow as tf
import tensorflow_addons as tfa
from matplotlib import pyplot as plt

dbcfg = dataset.Dataset()
val_data = dbcfg.load_val_data_with_annot()

image_paths = [i['imgpath'] for i in val_data]
bbox = [i['bbox'] for i in val_data]
joints = [i['joints'] for i in val_data]
scores = [i['score'] for i in val_data]

ds = tf.data.Dataset.from_tensor_slices((image_paths, bbox, joints, scores)).shuffle(100)
root_path = os.path.abspath('.')

# first step
data = []
for sample in ds.take(1):
    for i in sample:
        data.append(i)
        print(i.numpy())

# second step
sys.path.insert(0, osp.join(root_path, 'config'))
import config
config = config.config()

img = plt.imread(dbcfg.img_path + os.sep + data[0].numpy().decode("utf-8"))
cropped_img, target_coord = image_process.cropped_image_and_pose_coord(data[0],data[1], data[2])

# plt.imshow(img)
# plt.show()
# plt.imshow(cropped_img)
# plt.show()

# kps_array = np.array(data[2]).reshape((17, 3))
# print(kps_array.T)
# plt.imshow(dbcfg.vis_keypoints(img, kps_array.T))
# plt.show()

imaga_path = dbcfg.img_path

def data_process(file_path, bbox, joints, scores):
    [cropped_img, target_coord] = tf.py_function(image_process.cropped_image_and_pose_coord, 
                                        [file_path, bbox, joints], [tf.float32,  tf.int32])
    return cropped_img, target_coord

processed_ds = ds.shuffle(100).map(data_process).batch(32)

for images, coords in processed_ds.take(1):
    print('mini_batch images : ', images.shape)
    print('mini_batch labels : ', coords.shape)
    
plt.imshow(images[0].numpy().astype(int))
plt.show()