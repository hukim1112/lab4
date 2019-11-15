# 1. setting

### a) python library

This code work on TensorFlow 2.0 and few libraries for image processing such as Opencv

```
sudo apt-get install python3-tk

pip3 install tensorflow-gpu==2.0.0b1
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

# Implementation plan

## a) Simple pose estimation 

1. data pipeline with tf.data
2. build our network
3. connect person detector and our model



# Reference repository

1. Simple Baselines for Human Pose Estimation and Tracking [ [github](https://github.com/mks0601/TF-SimpleHumanPose) , [paper](https://arxiv.org/abs/1804.06208)]
2. Multiposenet [ [github](https://github.com/murdockhou/MultiPoseNet-tensorflow) , [paper](https://arxiv.org/abs/1807.04067) ]

