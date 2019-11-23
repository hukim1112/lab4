# 1. setting

### a) python library

This code work on TensorFlow 2.0 and few libraries for image processing such as Opencv

```
sudo apt-get install python3-tk

pip3 install tensorflow_addons
pip3 install tensorflow-gpu==2.0.0
pip3 install matplotlib
pip3 install opencv-python
pip3 install scikit-image
```

To do : release docker image for this project

### b) cocoapi for python (virtualenv recommended)

### your git will ignore files in cocoapi directory

```
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
pip install cython
python setup.py build_ext --inplace
python setup.py build_ext install
```

### c) Dataset path setting

Dataset path 설정해주기

```
${dataset_path}
|
|-- |-- COCO
|   `-- |-- dets
|       |   |-- human_detection.json
|       |-- annotations
|       |   |-- person_keypoints_train2017.json
|       |   |-- person_keypoints_val2017.json
|       |   `-- image_info_test-dev2017.json
|       `-- images
|           |-- train2017/
|           |-- val2017/
|           `-- test2017/
```

```
# config/coco_config.py에서 아래와 같이 image_path와 dataset_path를 수정해줄 것.

class config():
    def __init__(self):
        self.image_path = '{dataset_path}/COCO/images'
        self.dataset_path = '{dataset_path}/COCO/'
        self.input_shape = (256, 192)
        self.num_kps = 17
        self.rotation_factor = 40
        self.scale_factor = 0.3
        self.kps_symmetry = [(1, 2), (3, 4), (5, 6), (7, 8),
                             (9, 10), (11, 12), (13, 14), (15, 16)]
        self.output_shape = (
            self.input_shape[0] // 4, self.input_shape[1] // 4)
        if self.output_shape[0] == 64:
            self.sigma = 2
        elif self.output_shape[0] == 96:
            self.sigma = 3
        self.pixel_means = [[[123.68, 116.78, 103.94]]]
```

# Implementation plan

## a) Simple pose estimation 

1. ~~data pipeline with tf.data~~
2. ~~build our network~~
3. connect person detector and our model



# Reference repository

1. Simple Baselines for Human Pose Estimation and Tracking [ [github](https://github.com/mks0601/TF-SimpleHumanPose) , [paper](https://arxiv.org/abs/1804.06208)]
2. Multiposenet [ [github](https://github.com/murdockhou/MultiPoseNet-tensorflow) , [paper](https://arxiv.org/abs/1807.04067) ]

