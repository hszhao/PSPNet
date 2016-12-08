clear all; close all;

% change values here
is_server       = 1;

pos_w          = 3;
pos_x_std      = 3;

bi_w      = 3;    %5;
bi_x_std  = 95;   %50;
bi_r_std  = 3;    %10;

dataset = 'coco';  %'voc12', 'coco'

id         = 'comp6';
%trainset  = 'trainval_aug';
%trainset   = 'train_aug';
trainset   = 'train';

testset   = 'val';
%testset    = 'test';            %'val', 'test'

model_name = 'vgg128_noup_pool3';   %'vgg128_noup', 'vgg128_noup_glob', 'vgg128_ms'
feature_name = 'features';        %'features', 'features4', 'features2'
crf_feature_type = 'fc8_crf';   %'fc8', 'crf', 'fc8_crf'
fc8_feature_type = 'fc8';
save_feature_type = 'vis';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% You do not need to chage values below
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load('pascal_seg_colormap.mat');

if is_server
  if strcmp(dataset, 'voc12')
    VOC_root_folder = '/rmt/data/pascal/VOCdevkit';
  elseif strcmp(dataset, 'coco')
    VOC_root_folder = '/rmt/data/coco';
  else
    error('Wrong dataset');
  end
else
  if strcmp(dataset, 'voc12')  
    VOC_root_folder = '~/dataset/PASCAL/VOCdevkit';
  elseif strcmp(dataset, 'coco')
    VOC_root_folder = '~/dataset/coco';
  else
    error('Wrong dataset');
  end
end

crf_post_folder = sprintf('post_densecrf_W%d_XStd%d_RStd%d_PosW%d_PosXStd%d', bi_w, bi_x_std, bi_r_std, pos_w, pos_x_std); 
fc8_post_folder = 'post_none';

if strcmp(feature_name, 'features')
  crf_save_root_folder = fullfile('/rmt/work/deeplabel/exper', dataset, 'res', model_name, testset, crf_feature_type, crf_post_folder);
  fc8_save_root_folder = fullfile('/rmt/work/deeplabel/exper', dataset, 'res', model_name, testset, fc8_feature_type, fc8_post_folder);
  save_root_folder = fullfile('/rmt/work/deeplabel/exper', dataset, 'res', model_name, testset, save_feature_type);
else 
  crf_save_root_folder = fullfile('/rmt/work/deeplabel/exper', dataset, 'res', feature_name, model_name, testset, crf_feature_type, crf_post_folder);
  fc8_save_root_folder = fullfile('/rmt/work/deeplabel/exper', dataset, 'res', feature_name, model_name, testset, fc8_feature_type, fc8_post_folder);
  save_root_folder = fullfile('/rmt/work/deeplabel/exper', dataset, 'res', feature_name, model_name, testset, save_feature_type);
end

if ~exist(save_root_folder, 'dir')
    mkdir(save_root_folder);
end

if strcmp(dataset, 'voc12')
  crf_seg_res_dir = [crf_save_root_folder '/results/VOC2012/'];
  fc8_seg_res_dir = [fc8_save_root_folder '/results/VOC2012/'];
  gt_dir   = fullfile(VOC_root_folder, 'VOC2012', 'SegmentationClass');
elseif strcmp(dataset, 'coco')
  crf_seg_res_dir = [crf_save_root_folder '/results/COCO2014/'];
  fc8_seg_res_dir = [fc8_save_root_folder '/results/COCO2014/'];
  gt_dir   = fullfile(VOC_root_folder, '', 'SegmentationClass');
end

crf_save_result_folder = fullfile(crf_seg_res_dir, 'Segmentation', [id '_' testset '_cls']);
fc8_save_result_folder = fullfile(fc8_seg_res_dir, 'Segmentation', [id '_' testset '_cls']);


fc8_dir = dir(fullfile(fc8_save_result_folder, '*.png'));

for i = 1 : numel(fc8_dir)
    fprintf(1, 'processing %d (%d)...\n', i, numel(fc8_dir));
    
    fc8 = imread(fullfile(fc8_save_result_folder, fc8_dir(i).name));
    crf = imread(fullfile(crf_save_result_folder, fc8_dir(i).name));
    
    img_fn = fc8_dir(i).name(1:end-4);
    if strcmp(dataset, 'voc12')
      img = imread(fullfile(VOC_root_folder, 'VOC2012', 'JPEGImages', [img_fn, '.jpg']));
    elseif strcmp(dataset, 'coco')
      img = imread(fullfile(VOC_root_folder, 'JPEGImages', [img_fn, '.jpg']));
    end
    
    gt = imread(fullfile(gt_dir, [img_fn, '.png']));
    
    h = figure(1); 
    subplot(221),imshow(img), title('img');
    subplot(222),imshow(gt, colormap), title('gt');
    subplot(223), imshow(fc8, colormap), title('none');
    subplot(224), imshow(crf,colormap), title('crf');    
    
    save_fn = fullfile(save_root_folder, [img_fn '.jpg']);
    print(h, '-djpeg', save_fn);
end
