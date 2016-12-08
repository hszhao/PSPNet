% set up the environment variables
%

clear all; close all;
load('./pascal_seg_colormap.mat');

is_server       = 1;

crf_load_mat    = 1;   % the densecrf code load MAT files directly (no call SaveMatAsBin.m)
                       % used ONLY by DownSampleFeature.m
learn_crf       = 0;   % NOT USED. Set to 0

is_mat          = 1;   % the results to be evaluated are saved as mat (1) or png (0)
has_postprocess = 0;   % has done densecrf post processing (1) or not (0)
is_argmax       = 0;   % the output has been taken argmax already (e.g., coco dataset). 
                       % assume the argmax takes C-convention (i.e., start from 0)

debug           = 0;   % if debug, show some results

% vgg128_noup (not optimized well), aka DeepLab
% bi_w = 5, bi_x_std = 50, bi_r_std = 10

% vgg128_ms_pool3, aka DeepLab-MSc
% bi_w = 3, bi_x_std = 95, bi_r_std = 3

% vgg128_noup_pool3_cocomix, aka DeepLab-COCO
% bi_w = 5, bi_x_std = 67, bi_r_std = 3

%% these are used for the bounding box weak annotation experiments (i.e., to generate the Bbox-Seg)
% erode_gt (bbox)
% bi_w = 41, bi_x_std = 33, bi_r_std = 4

% erode_gt/bboxErode20
% bi_w = 45, bi_x_std = 37, bi_r_std = 3, pos_w = 15, pos_x_std = 3
 

%
% initial or default values for crf
bi_w           = 5; 
bi_x_std       = 50;
bi_r_std       = 3;

pos_w          = 3;
pos_x_std      = 3;


%
dataset    = 'voc12';  %'voc12', 'coco'
trainset   = 'train_aug';      % not used
testset    = 'val';            %'val', 'test'

model_name = 'vgg128_noup';

feature_name = 'features';
feature_type = 'fc8'; % fc8 / crf

id           = 'comp6';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% used for cross-validation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng(10)

% downsampling files for cross-validation
down_sample_method = 2;      % 1: equally sample with "down_sample_rate", 2: randomly pick "num_sample" samples
down_sample_rate   = 8;
num_sample         = 100;    % number of samples used for cross-validation

% ranges for cross-validation
range_pos_w = [3];
range_pos_x_std = [3];

range_bi_w = [5];
range_bi_x_std = [49];
range_bi_r_std = [4 5];


