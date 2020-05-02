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

#### Code coming soon.
