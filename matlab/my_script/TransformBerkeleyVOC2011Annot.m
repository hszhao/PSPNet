clear all; close all;

dataset = 'VOC2012';

%copy berkeley annotations
orig_folder = '../Berkeley_annot/dataset/cls';
save_folder = ['../', dataset, '/SegmentationClassAug_Visualization'];

if ~exist(save_folder, 'dir')
    mkdir(save_folder)
end

tmp = load('pascal_seg_colormap.mat');
colormap = tmp.colormap;

annots = dir(fullfile(orig_folder, '*.mat'));

for i = 1 : numel(annots)
    fprintf(1, 'processing %d (%d) ...\n', i, numel(annots));
    
    gt = load(fullfile(orig_folder, annots(i).name));
    
    imwrite(gt.GTcls.Segmentation, colormap, fullfile(save_folder, [annots(i).name(1:end-4), '.png']));
end

% copy pascal annotations
orig_folder = ['../', dataset, '/SegmentationClass'];
annots = dir(fullfile(orig_folder, '*.png'));

for i = 1 : numel(annots)
    fprintf(1, 'processing %d (%d) ...\n', i, numel(annots));
    
    gt = imread(fullfile(orig_folder, annots(i).name));
    
    imwrite(gt, colormap, fullfile(save_folder, annots(i).name));
end