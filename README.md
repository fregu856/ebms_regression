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

- TODO!









## Index
- [Usage](#usage)
- - [1D Regression](#1dregression)
- - [Object Detection](#detection)
- [Documentation](#documentation)
- - [1D Regression](#documentation1dregression)
- - [Object Detection](#documentationdetection)
- [Pretrained model](#pretrained-model)
***
***
***










***
***
***
## Usage

The code has been tested on Ubuntu 16.04. Docker images are provided (see below).

- [1D Regression](#1dregression)
- [Object Detection](#detection)






### 1dregression

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

- Example usage:
```
TODO!
```
***
***
***











### detection

- $ sudo docker pull fregu856/evaluating_bdl:rainbowsecret_pytorch04_20180905_evaluating_bdl
- Create _start_docker_image_segmentation.sh_ containing (My username on the server is _fregu482_, i.e., my home folder is _/home/fregu482_. You will have to modify this accordingly):
```      
#!/bin/bash

# DEFAULT VALUES
GPUIDS="0,1"
NAME="segmentation_GPU"

NV_GPU="$GPUIDS" nvidia-docker run -it --rm --shm-size 12G \
        -p 5900:5900 \
        --name "$NAME""01" \
        -v /home/fregu482:/home/ \
        fregu856/evaluating_bdl:rainbowsecret_pytorch04_20180905_evaluating_bdl bash
```
- (Inside the image, _/home/_ will now be mapped to _/home/fregu482_, i.e., $ cd home takes you to the regular home folder)
- (To create more containers, change the lines _GPUIDS="0,1"_, _--name "$NAME""01"_ and _-p 5900:5900_)
- General Docker usage:
- - To start the image:
- - - $ sudo sh start_docker_image_segmentation.sh
- - To commit changes to the image:
- - - Open a new terminal window.
- - - $ sudo docker commit segmentation_GPU01 fregu856/evaluating_bdl:rainbowsecret_pytorch04_20180905_evaluating_bdl
- - To exit the image without killing running code:
- - - Ctrl + P + Q
- - To get back into a running image:
- - - $ sudo docker attach segmentation_GPU01


- TODO!


- Example usage:
```
TODO!
```
***
***
***

















***
***
***
## Documentation

- [1D Regression](#documentation1dregression)
- [Object Detection](#documentationdetection)





### Documentation/1dregression

- Example usage:
```
TODO!
```

- TODO!
***
***
***















### Documentation/detection

- Example usage:
```
TODO!
```

- TODO!
***
***
***















***
***
***
## Pretrained model

- TODO!
