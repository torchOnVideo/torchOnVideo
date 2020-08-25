<p align="center"><img src="https://github.com/torchOnVideo/torchOnVideo/blob/combined_modules/docs/source/_static/img/torchOnVideoLogo.png" height="360px" /></p>

--------------------------------------------------------------------------------

torchOnVideo is a PyTorch based library for deep learning activities on videos.

It is an all encompassing library which provides models, training and testing strategies, custom data set classes and many more utilities involving state-of-the-art papers of the various subtasks of video based deep learning.

torchOnVideo aims to accelerate video deep learning research and interest and aims for the following:
* Simple and clean library interface
* Highly modular and well structured library
* Provides great functionality for both beginners and experts alike
* Access to metrics, data loaders and additional utils

**Overview**

* [`torchOnVideo.datasets`](https://github.com/torchOnVideo/torchOnVideo): Set of all custom dataset classes and utilities for video based datasets.
* [`torchOnVideo.frame_interpolation`](https://github.com/torchOnVideo/torchOnVideo): Models, classes and utilities related to video frame interpolation.
* [`torchOnVideo.super_resolution`](https://github.com/torchOnVideo/torchOnVideo): Models, classes and utilities related to video frame interpolation.
* [`torchOnVideo.denoising`](https://github.com/torchOnVideo/torchOnVideo): Models, classes and utilities related to video denoising.
* [`torchOnVideo.tracking`](https://github.com/torchOnVideo/torchOnVideo): Models, classes and utilities related to multi object tracking.
* [`torchOnVideo.video_captioning`](https://github.com/torchOnVideo/torchOnVideo):Models, classes and utilities related to video captioning.
* [`torchOnVideo.losses`](https://github.com/torchOnVideo/torchOnVideo): Utilities and losses for video based deep learning.
* [`torchOnVideo.metrics`](https://github.com/torchOnVideo/torchOnVideo): Utilities and metrics for video based deep learning.

**Resources**

* Website: *Coming very soon!*
* GitHub: [https://github.com/torchOnVideo/torchOnVideo](https://github.com/learnables/learn2learn/)


## Installation

~~~bash
python setup.py bdist_wheel
pip install -e torchOnVideo/dist/torchOnVideo-0.1.0-py3-none-any.whl
~~~

## Current subtasks and papers supported
*The paper name is in brackets*
* Frame Interpolation (CAIN, AdaCoF)
* Video SuperResolution (iSeeBetter, SOFVSR)
* Video denoising(VNLNet)
* Multiple Object Tracking (Fair MOT, Towards Real Time MOT) 
* Video Captioning (SALSTM)


## Snippets & Examples


The following snippets provide a sneak peek at the functionalities of torchOnVideo.

For all the cases we want to run the task of frame interpolation using the technique explained in the paper **Adaptive Collaboration of Flows for Video Frame Interpolation - AdaCoF net for short**, a state of the art paper by the researchers at the Yonsei university

### Train and save the exact training strategy as in the paper (default strategy):

~~~python
# call TrainModel class from adacof submodule of frame_interpolation
from torchOnVideo.frame_interpolation.adacof import TrainModel
adacof_obj = TrainModel()
adacof_obj()
~~~

### Train with another model on AdaCoF:

Here we demonstrate the flexibility of our library. Here, we are plugging in another model called MyNet and passing it to the TrainModel class
~~~python
# Training AdaCoF with own model
from mynet import MyNet
adacof_obj2 = TrainModel(model=MyNet)
adacof_obj2()
~~~

Likewise, one can change the dataset, dataloader, scheduler, optimizer and many more options.

### Test on AdaCoF:

~~~python
# Using the testing functionality
from torchOnVideo.frame_interpolation.adacof import TestModel
adacof_obj_test = TestModel()
adacof_obj_test(1)
~~~
The flexibility for TestModel is exactly the same as in TrainModel


### Directly using a model:
Suppose the user want to access the model(s) used in the paper. I can easily import the AdaCoFNet model and import it in my own code.

~~~python
# Directly using AdaCoFNet Model
from torchOnVideo.frame_interpolation.models import AdaCoFNet
kernel_size = 5
dilation = 1
model = AdaCoFNet(kernel_size, dilation)
~~~

### Using our provided custom dataset classes:
For eg if one wants to use the TrainAdaCoF dataset class built in the *Vimeo90K Triplet* dataset
~~~python
# Using the custom dataset class of AdaCoF
from torchOnVideo.datasets.Vimeo90KTriplet.frame_interpolation import TrainAdaCoF
train_set = TrainAdaCoF()
~~~




## References:
https://github.com/myungsub/CAIN

https://github.com/HyeongminLEE/AdaCoF-pytorch

https://github.com/LongguangWang/SOF-VSR

https://github.com/amanchadha/iSeeBetter

https://github.com/axeldavy/vnlnet

https://github.com/learnables/learn2learn/blob/master/README.md

https://github.com/Zhongdao/Towards-Realtime-MOT

https://github.com/ifzhang/FairMOT

https://github.com/hobincar/SA-LSTM

https://github.com/MDSKUL/MasterProject







