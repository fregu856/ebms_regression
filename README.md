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

### Code is coming soon!

***
***
***
***
***
***
***
***
***
***
***
***
***
***
***
***
***
***
***
***
***
***
***
***
***
***
***
***
***
***
***
***
***
***
***
***



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

The code has been tested on Ubuntu 16.04. A docker image is provided (see below).

- [1D Regression](#1dregression)
- [Object Detection](#detection)






### 1dregression

- $ docker pull fregu856/ebms_regression:ufoym_deepo_pytorch-py36-cu90_ebms_regression
- Create _start_docker_image_ebms_regression.sh_ containing (My username on the server is _fregu482_, i.e., my home folder is _/home/fregu482_. You will have to modify this accordingly):
```
#!/bin/bash

# DEFAULT VALUES
GPUIDS="0"
NAME="ebms_regression_GPU"

NV_GPU="$GPUIDS" nvidia-docker run -it --rm --shm-size 12G \
        -p 7200:7200\
        --name "$NAME""0" \
        -v /home/fregu482:/root/ \
        fregu856/ebms_regression:ufoym_deepo_pytorch-py36-cu90_ebms_regression bash
```
- (Inside the image, _/root/_ will now be mapped to _/home/fregu482_, i.e., $ cd -- takes you to the regular home folder)
- (To create more containers, change the lines _GPUIDS="0"_, _--name "$NAME""0"_ and _-p 7200:7200_)
- General Docker usage:
- - To start the image:
- - - $ sh start_docker_image_ebms_regression.sh
- - To commit changes to the image:
- - - Open a new terminal window.
- - - $ docker commit ebms_regression_GPU0 fregu856/ebms_regression:ufoym_deepo_pytorch-py36-cu90_ebms_regression
- - To exit the image without killing running code:
- - - Ctrl + P + Q
- - To get back into a running image:
- - - $ docker attach ebms_regression_GPU0

- Example usage:
```
$ sh start_docker_image_ebms_regression.sh
$ cd --
$ python ebms_regression/1dregression/1/nce+_train.py 
```
***
***
***











### detection

- $ docker pull fregu856/ebms_regression:ufoym_deepo_pytorch-py36-cu90_ebms_regression
- Create _start_docker_image_ebms_regression.sh_ containing (My username on the server is _fregu482_, i.e., my home folder is _/home/fregu482_. You will have to modify this accordingly):
```
#!/bin/bash

# DEFAULT VALUES
GPUIDS="0"
NAME="ebms_regression_GPU"

NV_GPU="$GPUIDS" nvidia-docker run -it --rm --shm-size 12G \
        -p 7200:7200\
        --name "$NAME""0" \
        -v /home/fregu482:/root/ \
        fregu856/ebms_regression:ufoym_deepo_pytorch-py36-cu90_ebms_regression bash
```
- (Inside the image, _/root/_ will now be mapped to _/home/fregu482_, i.e., $ cd -- takes you to the regular home folder)
- (To create more containers, change the lines _GPUIDS="0"_, _--name "$NAME""0"_ and _-p 7200:7200_)
- General Docker usage:
- - To start the image:
- - - $ sh start_docker_image_ebms_regression.sh
- - To commit changes to the image:
- - - Open a new terminal window.
- - - $ docker commit ebms_regression_GPU0 fregu856/ebms_regression:ufoym_deepo_pytorch-py36-cu90_ebms_regression
- - To exit the image without killing running code:
- - - Ctrl + P + Q
- - To get back into a running image:
- - - $ docker attach ebms_regression_GPU0


- TODO! (download datasets and pretrained detector)


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
$ sh start_docker_image_ebms_regression.sh
$ cd --
$ python ebms_regression/1dregression/1/nce+_train.py 
```

1dregression/1 contains all code for the first dataset, 1dregression/2 all code for the second dataset.

- _1dregression/1/model.py_: Definition of the feed-forward DNN f_\theta(x, y). Identical to _1dregression/2/model.py_.
- _1dregression/{1, 2}/datasets.py_: Definition of the {first, second} dataset.
- _1dregression/{1, 2}/{{mlis, mlmcmcL16, kldis, nce, sm, dsm, nce+}}_train.py_: Train 20 models on the {first, second} dataset using {{ML-IS, ML-MCMC-16, KLD-IS, NCE, SM, DSM, NCE+}}.
- _ 1dregression/{1, 2}/{{mlis, mlmcmcL16, kldis, nce, sm, dsm, nce+}}_eval.py_: Evaluate the KL divergence to the true p(y | x) for all 20 trained models, compute the mean for the 5 best models. 

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
