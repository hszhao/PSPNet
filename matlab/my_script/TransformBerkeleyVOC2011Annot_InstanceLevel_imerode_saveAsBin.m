% use instance-level information to generate "bbox" segmentation map
% the bbox is eroded
% Directly save the result as bin files

clear all; close all;

debug = 0;
dataset = 'VOC2012';
is_server = 1;

do_berkeley = 1;
do_pascal   = 1;

% erode the bbox to be (erode_side_scale)% of the original bbox
erode_scale    = 0.2;       % 1: no erosion
soft_erosion   = 0;          % 1: soft, 0: hard
occulder_bias = 0.1;     % amount of occulder bias. Use 0.25 makes one +0.05 and another one -0.05

%copy berkeley annotations
if soft_erosion
    prefix = 'Soft';
else
    prefix = '';
end

if occulder_bias ~= 0
    postfix = '_OccluBias';
else
    postfix = '';
end

if is_server
    save_folder = fullfile('/rmt/work/deeplabel/exper/voc12/erode_gt', sprintf(['bbox' prefix 'Erode%2d' postfix], erode_scale*100));
    voc_root      = '/rmt/data/pascal/VOCdevkit';
else
    save_folder = fullfile('~/workspace/deeplabeling/exper/voc12/erode_gt', sprintf(['bbox' prefix 'Erode%2d' postfix], erode_scale*100));
    voc_root      = '~/dataset/PASCAL/VOCdevkit';
    img_folder =fullfile( '~/dataset/PASCAL/VOCdevkit', dataset,  'JPEGImages');
end

if ~exist(save_folder, 'dir')
    mkdir(save_folder)
end

if strcmp(dataset, 'VOC2012')
    num_class = 21;
end

tmp = load('pascal_seg_colormap.mat');
colormap = tmp.colormap;

if do_berkeley
    orig_cls_folder  = fullfile(voc_root, 'Berkeley_annot', 'dataset', 'cls');
    orig_inst_folder = fullfile(voc_root, 'Berkeley_annot', 'dataset', 'inst');

    cls_annots = dir(fullfile(orig_cls_folder, '*.mat'));

    for i = 1 : numel(cls_annots)
        fprintf(1, 'processing %d (%d) ...\n', i, numel(cls_annots));

        cls_gt  = load(fullfile(orig_cls_folder, cls_annots(i).name));
        inst_gt = load(fullfile(orig_inst_folder, cls_annots(i).name));

        classes = [0; inst_gt.GTinst.Categories];  %append bkg class
        seg_bbox = PasteSoftSegmentBboxByArea(inst_gt.GTinst.Segmentation, classes, num_class, 'descend', erode_scale, soft_erosion, occulder_bias, debug);

        save_fn = fullfile(save_folder, [cls_annots(i).name(1:end-4), '.bin']);
        SaveBinFile(seg_bbox, save_fn, 'float');

        if debug
            img = imread(fullfile(img_folder, [cls_annots(i).name(1:end-4), '.jpg']));
            figure(1)
            subplot(2,2,1), imshow(uint8(cls_gt.GTcls.Segmentation), colormap), title('cls')
            subplot(2,2,2), imshow(uint8(inst_gt.GTinst.Segmentation), colormap), title('inst')
            subplot(2,2,3), imshow(img), title('img');
            [~, seg_label] = max(seg_bbox, [], 3);
            subplot(2,2,4), imshow(uint8(seg_label-1), colormap), title('seg bbox')
            pause();
        end
    end
end

% copy pascal annotations
if do_pascal

    orig_cls_folder  = fullfile(voc_root, dataset, 'SegmentationClass');
    orig_inst_folder = fullfile(voc_root, dataset, 'SegmentationObject');

    cls_annots = dir(fullfile(orig_cls_folder, '*.png'));

    for i = 1 : numel(cls_annots)
        fprintf(1, 'processing %d (%d) ...\n', i, numel(cls_annots));

        cls_gt = imread(fullfile(orig_cls_folder, cls_annots(i).name));
        inst_gt = imread(fullfile(orig_inst_folder, cls_annots(i).name));

        % append background 0
        inst_categories = unique([ [0; inst_gt(:)], [0; cls_gt(:)]], 'rows');
        categories = inst_categories(:,2);

        seg_bbox = PasteSoftSegmentBboxByArea(inst_gt, categories, num_class, 'descend', erode_scale, soft_erosion, occulder_bias, debug);

        save_fn = fullfile(save_folder, [cls_annots(i).name(1:end-4), '.bin']);
        SaveBinFile(seg_bbox, save_fn, 'float');

        if debug
            img = imread(fullfile(img_folder, [cls_annots(i).name(1:end-4), '.jpg']));
            figure(2)
            subplot(2,2,1), imshow(uint8(cls_gt), colormap), title('cls')
            subplot(2,2,2), imshow(uint8(inst_gt), colormap), title('inst')
            subplot(2,2,3), imshow(img), title('img');
            [~, seg_label] = max(seg_bbox, [], 3) ;
            subplot(2,2,4), imshow(uint8(seg_label-1), colormap), title('seg bbox')
            pause();
        end

    end
end
