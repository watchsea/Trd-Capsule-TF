# Trd-Capsule-TF

[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=plastic)](CONTRIBUTING.md)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg?style=plastic)](https://opensource.org/licenses/Apache-2.0)
![completion](https://img.shields.io/badge/completion%20state-90%25-blue.svg?style=plastic)
[![Gitter](https://img.shields.io/gitter/room/nwjs/nw.js.svg?style=plastic)](https://gitter.im/CapsNet-Tensorflow/Lobby)

A Tensorflow implementation of CapsNet based on Geoffrey Hinton's paper [Dynamic Routing Between Capsules](https://arxiv.org/abs/1710.09829)

> **Status:**
> 1. The capsule of MNIST version is finished. Now we have two version:

> **Daily task**
> 1. multi-GPU support
> 2. Improving the reusability of ``capsLayer.py``, what you need is ``import capsLayer.fully_connected`` or ``import capsLayer.conv2d`` in your code



## Requirements
- Python
- NumPy
- [Tensorflow](https://github.com/tensorflow/tensorflow) (I'm using 1.3.0, not yet tested for older version)
- tqdm (for displaying training progress info)
- scipy (for saving images)

## Author Info
QQ:328485771




------------

 
- There is [another new paper](https://openreview.net/pdf?id=HJWLfGWRb) about capsules(submitted to ICLR 2018), a follow-up of the CapsNet paper.

Thanks the authors of the following:
https://github.com/naturomics/CapsNet-Tensorflow
https://github.com/XifengGuo/CapsNet-Keras
https://github.com/nishnik/CapsNet-PyTorch
