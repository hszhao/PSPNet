 % evaluate the pixel accuracy along object boundary
 %
 
 clear all; close all;

% change values here
boundary_w = 1:40;
dataset         = 'VOC2012';

test_berkeley      = 0;
is_server             = 0;
has_postprocess = 1;   % has done densecrf post processing or not

pos_w          = 3;
pos_x_std      = 3;

bi_w            = 5;
bi_x_std      = 50;
bi_r_std       = 10;

id         = 'comp6';
%trainset  = 'trainval_aug';
trainset   = 'train_aug';

%testset   = 'trainval_aug';
testset    = 'val';

model_name = 'vgg128_noup';   %'vgg128_noup', 'vgg128_noup_glob', 'vgg128_ms_pool3'

if has_postprocess
  post_folder = sprintf('post_densecrf_W%d_XStd%d_RStd%d_PosW%d_PosXStd%d', bi_w, bi_x_std, bi_r_std, pos_w, pos_x_std); 
else
  post_folder = 'post_none';
end

if is_server
    VOC_root_folder = '/rmt/data/pascal/VOCdevkit';
    save_root_folder = fullfile('/rmt/work/deeplabel/exper/voc12/res', model_name, testset, 'fc8', post_folder);
else
    VOC_root_folder = '~/dataset/PASCAL/VOCdevkit';
    save_root_folder = fullfile('~/workspace/deeplabeling/exper/voc12/res', model_name, testset, 'fc8', post_folder);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% You do not need to chage values below
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pixel_acc = zeros(1, length(boundary_w));
class_acc = zeros(1, length(boundary_w));
pixel_iou = zeros(1, length(boundary_w));

seg_res_dir = [save_root_folder '/results/VOC2012/'];
save_result_folder = fullfile(seg_res_dir, 'Segmentation', [id '_' testset '_cls']);

seg_root = fullfile(VOC_root_folder, dataset);

if ~exist(save_result_folder, 'dir')
    mkdir(save_result_folder);
end

if test_berkeley
    seg_res_dir = '~/workspace/deeplabeling/Berkeley_FCN_results/results_color/val2011/fcn-8s/';
end

VOCopts = GetVOCopts(seg_root, seg_res_dir, trainset, testset, dataset);

for i = 1 : length(boundary_w)
    w = boundary_w(i);
    
    % get iou score
    [accuracies, avacc, conf, rawcounts, p_acc, c_acc] = MyVOCevalsegBoundary(VOCopts,  id,  w);
    
    pixel_acc(i) = p_acc;
    class_acc(i) = c_acc;
    pixel_iou(i) = avacc;
end