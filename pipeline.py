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

config = config()
dbcfg = dataset.Dataset()
train_data = dbcfg.load_val_data_with_annot()


image_paths = [i['imgpath'] for i in train_data]
bbox = [i['bbox'] for i in train_data]
joints = [i['joints'] for i in train_data]

ds = tf.data.Dataset.from_tensor_slices(
    (image_paths, bbox, joints)).shuffle(10000)


def data_process(file_path, bbox, joints):
    [cropped_img, target_coord] = tf.py_function(image_process.cropped_image_and_pose_coord,
                                                 [file_path, bbox, joints], [tf.float32, tf.int16])
    heatmap = image_process.render_gaussian_heatmap(
        target_coord, config.output_shape)
    return cropped_img, heatmap


processed_ds = ds.map(data_process).batch(32)

#pose_estimator = tf.keras.models.load_model('model/pose.h5')

for i in processed_ds.take(1):
    print('input images : ', i[0].shape) #input images :  (32, 256, 192, 3)
    print('heatmap (label) shape : ', i[1].shape) #heatmap (label) shape :  (32, 64, 48, 17)
    #print('predicted heatmap shape : ', pose_estimator(i[0]).shape) # predicted heatmap shape :  (32, 64, 48, 17)
