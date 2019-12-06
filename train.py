import os
import os.path as osp
import sys
import numpy as np
import cv2
from data_util.COCO import dataset, image_process
import tensorflow as tf
from matplotlib import pyplot as plt
from config.coco_config import config
import gpu_handle

gpu_handle.setup_gpus()

config = config()
dbcfg = dataset.Dataset()
train_data = dbcfg.load_train_data()
val_data =dbcfg.load_val_data_with_annot()

image_paths = [i['imgpath'] for i in train_data]
bbox = [i['bbox'] for i in train_data]
joints = [i['joints'] for i in train_data]

ds = tf.data.Dataset.from_tensor_slices(
    (image_paths, bbox, joints)).shuffle(10000)

image_paths = [i['imgpath'] for i in val_data]
bbox = [i['bbox'] for i in val_data]
joints = [i['joints'] for i in val_data]

val_ds = tf.data.Dataset.from_tensor_slices(
    (image_paths, bbox, joints))


def fixup_shape(cropped_img, heatmap):
    cropped_img.set_shape([256, 192, 3])
    heatmap.set_shape([64, 48, 17])  # I have 17 classes
    return cropped_img, heatmap
def data_process(file_path, bbox, joints):
    [cropped_img, target_coord] = tf.py_function(image_process.cropped_image_and_pose_coord, [file_path, bbox, joints], [tf.float32, tf.int16])
    heatmap = image_process.render_gaussian_heatmap(target_coord, config.output_shape)
    return fixup_shape(cropped_img/255., heatmap)


processed_ds = ds.map(data_process).batch(48).repeat()
processed_val_ds = ds.map(data_process).batch(64)
pose_estimator = tf.keras.models.load_model('model/pose.h5')

lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=40000,
        decay_rate = 0.96,
        staircase=True)
optimizer = tf.keras.optimizers.Adam(lr_schedule)
pose_estimator.compile(loss='mse',
                       optimizer=optimizer,
                       metrics=['mae', 'mse'])

EPOCHS = 120

callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath='model/trained_{epoch}.h5',
            save_best_only = True,
            monitor='val_loss',
            verbose=1)
        ]

history = pose_estimator.fit(
    processed_ds,
    validation_data=processed_val_ds,
    callbacks=callbacks,
    epochs=EPOCHS)

pose_estimator.save('trained.h5')
