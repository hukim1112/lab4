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
train_data = dbcfg.load_train_data()

image_paths = [i['imgpath'] for i in train_data]
bbox = [i['bbox'] for i in train_data]
joints = [i['joints'] for i in train_data]

ds = tf.data.Dataset.from_tensor_slices(
    (image_paths, bbox, joints)).shuffle(10000)


def fixup_shape(cropped_img, heatmap):
    cropped_img.set_shape([None, 256, 192, 3])
    heatmap.set_shape([None, 64, 48, 17])  # I have 17 classes
    return cropped_img, heatmap
def data_process(file_path, bbox, joints):
    [cropped_img, target_coord] = tf.py_function(image_process.cropped_image_and_pose_coord, [file_path, bbox, joints], [tf.float32, tf.int16])
    heatmap = image_process.render_gaussian_heatmap(target_coord, config.output_shape)
    return fixup_shape(cropped_img/255., heatmap)


processed_ds = ds.map(data_process).batch(64)
pose_estimator = tf.keras.models.load_model('model/pose.h5')
optimizer = tf.keras.optimizers.Adam(0.001)
pose_estimator.compile(loss='mse',
                       optimizer=optimizer,
                       metrics=['mae', 'mse'])

EPOCHS = 120

history = pose_estimator.fit(
    processed_ds,
    epochs=EPOCHS)
