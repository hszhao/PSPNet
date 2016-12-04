## Pyramid Scene Parsing Network

by Hengshuang Zhao, Jianping Shi, Xiaojuan Qi, Xiaogang Wang, Jiaya Jia, details are in [project page](http://www.cse.cuhk.edu.hk/~hszhao/projects/pspnet/index.html).

### Introduction

This repository is modified from Caffe version of [yjxiong](https://github.com/yjxiong/caffe/tree/mem) and [DeepLab v2](https://bitbucket.org/aquariusjay/deeplab-public-ver2) for testing. Results are tested on Ubuntu 14.04. Trainable code will be available later.

### Usage

1. Clone the repository:

   ```shell
   git clone https://github.com/hszhao/PSPNet.git
   ```

2. Build Caffe and matcaffe

   ```shell
   cd $PSPNET_ROOT
   make -j8 && make matcaffe
   ```

3. Testing

   Evaluation code is in folder 'evaluation'.

   Download trained models and put it in folder 'evaluation/model/':

   pspnet50_ADE20K.caffemodel: [link](https://drive.google.com/open?id=0BzaU285cX7TCN1R3QnUwQ0hoMTA)

   pspnet101_VOC2012.caffemodel: [link](https://drive.google.com/open?id=0BzaU285cX7TCNVhETE5vVUdMYk0)

   pspnet101_cityscapes.caffemodel: [link](https://drive.google.com/open?id=0BzaU285cX7TCT1M3TmNfNjlUeEU)

   Modify the related paths in 'eval_all.m':

   ```shell
   cd evaluation
   vim eval_all.m
   ```

   Run the testing scripts:

   ```
   ./run.sh
   ```

4. Results: (single scale testing denotes as 'ss' and multiple scale testing denotes as 'ms')

   - PSPNet50 on ADE20K valset (mIoU/pAcc): 41.68/80.04 (ss) and 42.78/80.76 (ms) 
   - PSPNet101 on VOC2012 testset (mIoU): 85.41 (ms)
   - PSPNet101 on cityscapes valset (mIoU/pAcc): 79.70/96.38 (ss) and 80.91/96.59 (ms)

### Questions

Please contact 'hszhao@cse.cuhk.edu.hk'
