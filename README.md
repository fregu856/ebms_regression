# ebms_regression

![overview image](ebms_regression.jpg)

Official implementation (PyTorch) of the paper: \
**How to Train Your Energy-Based Model for Regression**, 2020 [[arXiv (TODO!)]]() [[project (TODO!)]](). \
_[Fredrik K. Gustafsson](http://www.fregu856.com/), [Martin Danelljan](https://martin-danelljan.github.io/), [Radu Timofte](http://people.ee.ethz.ch/~timofter/), [Thomas B. Schön](http://user.it.uu.se/~thosc112/)._ 

Energy-based models (EBMs) have become increasingly popular within computer vision in recent years. While they are commonly employed for generative image modeling, recent work has applied EBMs also for regression tasks, achieving state-of-the-art performance on object detection and visual tracking. Training EBMs is however known to be challenging. While a variety of different techniques have been explored for generative modeling, the application of EBMs to regression is not a well-studied problem. How EBMs should be trained for best possible regression performance is thus currently unclear. We therefore accept the task of providing the first detailed study of this problem. To that end, we propose a simple yet highly effective extension of noise contrastive estimation, and carefully compare its performance to six popular methods from literature on the tasks of 1D regression and object detection. The results of this comparison suggest that our training method should be considered the go-to approach. We also apply our method to the visual tracking task, setting a new state-of-the-art on five datasets. Notably, our tracker achieves 63.7% AUC on LaSOT and 78.7% Success on TrackingNet.

This repository contains code for the experiments on _**object detection**_ and _**1D regression**_. Code for the _**visual tracking**_ experiments is available at [pytracking](https://github.com/visionml/pytracking).

If you find this work useful, please consider citing:
```
TODO!
```

### Code coming soon.



## Acknowledgements

- The depthCompletion code is based on the implementation by [@fangchangma](https://github.com/fangchangma) found [here](https://github.com/fangchangma/self-supervised-depth-completion).
- The segmentation code is based on the implementation by [@PkuRainBow](https://github.com/PkuRainBow) found [here](https://github.com/PkuRainBow/OCNet.pytorch), which in turn utilizes [inplace_abn](https://github.com/mapillary/inplace_abn) by [@mapillary](https://github.com/mapillary).









## Index
- [Usage](#usage)
- - [depthCompletion](#depthcompletion)
- - [segmentation](#segmentation)
- - [toyRegression](#toyregression)
- - [toyClassification](#toyclassification)
- [Documentation](#documentation)
- - [depthCompletion](#documentationdepthcompletion)
- - [segmentation](#documentationsegmentation)
- - [toyRegression](#documentationtoyregression)
- - [toyClassification](#documentationtoyclassification)
- [Pretrained models](#pretrained-models)
***
***
***










***
***
***
## Usage

The code has been tested on Ubuntu 16.04. Docker images are provided (see below).

- [depthCompletion](#depthcompletion)
- [segmentation](#segmentation)
- [toyRegression](#toyregression)
- [toyClassification](#toyclassification)






### depthCompletion

- $ sudo docker pull fregu856/evaluating_bdl:pytorch_pytorch_0.4_cuda9_cudnn7_evaluating_bdl
- Create _start_docker_image_toyProblems_depthCompletion.sh_ containing (My username on the server is _fregu482_, i.e., my home folder is _/home/fregu482_. You will have to modify this accordingly):
```
#!/bin/bash

# DEFAULT VALUES
GPUIDS="0"
NAME="toyProblems_depthCompletion_GPU"

NV_GPU="$GPUIDS" nvidia-docker run -it --rm --shm-size 12G \
        -p 5700:5700\
        --name "$NAME""0" \
        -v /home/fregu482:/root/ \
        fregu856/evaluating_bdl:pytorch_pytorch_0.4_cuda9_cudnn7_evaluating_bdl bash
```
- (Inside the image, _/root/_ will now be mapped to _/home/fregu482_, i.e., $ cd -- takes you to the regular home folder)
- (To create more containers, change the lines _GPUIDS="0"_, _--name "$NAME""0"_ and _-p 5700:5700_)
- General Docker usage:
- - To start the image:
- - - $ sudo sh start_docker_image_toyProblems_depthCompletion.sh
- - To commit changes to the image:
- - - Open a new terminal window.
- - - $ sudo docker commit toyProblems_depthCompletion_GPU0 fregu856/evaluating_bdl:pytorch_pytorch_0.4_cuda9_cudnn7_evaluating_bdl
- - To exit the image without killing running code:
- - - Ctrl + P + Q
- - To get back into a running image:
- - - $ sudo docker attach toyProblems_depthCompletion_GPU0



- Download the [KITTI depth completion](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion) dataset (_data_depth_annotated.zip_, _data_depth_selection.zip_ and _data_depth_velodyne.zip_) and place it in _/root/data/kitti_depth_ (_/root/data/kitti_depth_ should contain the folders _train_, _val_ and _depth_selection_).

- Create _/root/data/kitti_raw_ and download the [KITTI raw](http://www.cvlibs.net/datasets/kitti/raw_data.php) dataset using [download_kitti_raw.py](https://github.com/fregu856/evaluating_bdl/blob/master/depthCompletion/utils/download_kitti_raw.py).

- Create _/root/data/kitti_rgb_. For each folder in _/root/data/kitti_depth/train_ (e.g. _2011_09_26_drive_0001_sync_), copy the corresponding folder in _/root/data/kitti_raw_ and place it in _/root/data/kitti_rgb/train_.


- Download the [virtual KITTI](https://europe.naverlabs.com/Research/Computer-Vision/Proxy-Virtual-Worlds/) dataset (_vkitti_1.3.1_depthgt.tar_ and _vkitti_1.3.1_rgb.tar_) and place in _/root/data/virtualkitti_ (_/root/data/virtualkitti_ should contain the folders _vkitti_1.3.1_depthgt_ and _vkitti_1.3.1_rgb_).



- Example usage:
```
$ sudo sh start_docker_image_toyProblems_depthCompletion.sh
$ cd --
$ python evaluating_bdl/depthCompletion/ensembling_train_virtual.py
```
***
***
***
