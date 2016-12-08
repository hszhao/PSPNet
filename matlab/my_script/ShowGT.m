tmp = load('pascal_seg_colormap.mat');
colormap = tmp.colormap;

fn = '2007_000032.png';
gt1 = imread(fullfile('../SegmentationClass', fn));
gt2 = imread(fullfile('../SegmentationClassAug', fn));
figure(1), subplot(121), imshow(gt1, colormap), subplot(122), imshow(gt2, colormap)