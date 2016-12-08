## Pyramid Scene Parsing Network

by Hengshuang Zhao, Jianping Shi, Xiaojuan Qi, Xiaogang Wang, Jiaya Jia, details are in [project page](http://www.cse.cuhk.edu.hk/~hszhao/projects/pspnet/index.html).

### Introduction

This repository is modified from Caffe version of [yjxiong](https://github.com/yjxiong/caffe/tree/mem) and [DeepLab v2](https://bitbucket.org/aquariusjay/deeplab-public-ver2) for evaluation. We merge the batch normalization layer named 'bn_layer' in the former one into the later one while keep the original 'batch_norm_layer' in the later one unchanged for compatibility. The difference is that 'bn_layer' contains four parameters as 'slope,bias,mean,variance' while 'batch_norm_layer' contains two parameters as 'mean,variance'. Several evaluation code is borrowed from [MIT Scene Parsing](https://github.com/CSAILVision/sceneparsing).

## Installation

For installation, please follow the instructions of [Caffe](https://github.com/BVLC/caffe) and [DeepLab v2](https://bitbucket.org/aquariusjay/deeplab-public-ver2). To enable cuDNN for GPU acceleration, cuDNN v5 is needed as required in 'yjxiong'. If you meet error related with 'matio', please download and install [matio](matio) as required in 'DeepLab v2'.

The code has been tested successfully on Ubuntu 14.04 and 12.04 with CUDA 7.0.

### Usage

1. Clone the repository:

   ```shell
   git clone https://github.com/hszhao/PSPNet.git
   ```

2. Build Caffe and matcaffe:

   ```shell
   cd $PSPNET_ROOT
   cp Makefile.config.example Makefile.config
   vim Makefile.config
   make -j8 && make matcaffe
   ```

3. Evaluation:

   - Evaluation code is in folder 'evaluation'.
   - Download trained models and put them in folder 'evaluation/model':
     - pspnet50\_ADE20K.caffemodel: [GoogleDrive](https://drive.google.com/open?id=0BzaU285cX7TCN1R3QnUwQ0hoMTA)
     - pspnet101\_VOC2012.caffemodel: [GoogleDrive](https://drive.google.com/open?id=0BzaU285cX7TCNVhETE5vVUdMYk0)
     - pspnet101\_cityscapes.caffemodel: [GoogleDrive](https://drive.google.com/open?id=0BzaU285cX7TCT1M3TmNfNjlUeEU)
   - Modify the related paths in 'eval_all.m':
     - Mainly variables 'data_root' and 'eval_list', and your image list for evaluation should be similarity to that in folder 'evaluation/samplelist' if you use this evaluation code structure. 
     - Matlab 'parfor' evaluation is used and the default GPUs are with ID [0:3]. Modify variable 'gpu_id_array' if needed. We assume that number of images can be divided by number of GPUs; if not, you can just pad your image list or switch to single GPU evaluation by set 'gpu_id_array' be length of one, and change 'parfor' to 'for' loop.

   ```shell
   cd evaluation
   vim eval_all.m
   ```

   - Run the evaluation scripts:

   ```
   ./run.sh
   ```

4. Results: 

   Prediction results will show in folder 'evaluation/mc_result' and the expected scores are:

   (single scale testing denotes as 'ss' and multiple scale testing denotes as 'ms')

   - PSPNet50 on ADE20K valset (mIoU/pAcc): 41.68/80.04 (ss) and 42.78/80.76 (ms) 
   - PSPNet101 on VOC2012 testset (mIoU): 85.41 (ms)
   - PSPNet101 on cityscapes valset (mIoU/pAcc): 79.70/96.38 (ss) and 80.91/96.59 (ms)

5. Demo video:

   Video processed by PSPNet101 on cityscapes dataset: [Youtube](https://youtu.be/gdAVqJn_J2M)

## Citation

If PSPNet is useful for your research, please consider citing:

    @article{zhao2016pspnet,
      author    = {Hengshuang Zhao and
      			   Jianping Shi and
      			   Xiaojuan Qi and
      			   Xiaogang Wang and
      			   Jiaya Jia},
      title     = {Pyramid Scene Parsing Network},
      journal   = {CoRR},
      volume    = {abs/1612.01105},
      year      = {2016},
      url       = {https://arxiv.org/abs/1612.01105},
    }
### Questions

Please contact 'hszhao@cse.cuhk.edu.hk'
