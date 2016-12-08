%function GetImglistForCaffe()

data_folder = '../VOC2012/ImageSets/Segmentation';
save_folder = '~/workspace/caffe-dev/examples/segnet';

img_prefix = '/JPEGImages/';
img_postfix = '.jpg';
seg_prefix = '/SegmentationClassAug/';
seg_postfix = '.png';

fn = {'VOC2012_test.txt', ...
         'VOC2012_train.txt', ...
          'VOC2012_val.txt', ...
          'VOC2012_train_aug.txt', ...
          'VOC2012_trainval_aug.txt'};
    
for i = 1 : numel(fn)
    list = GetList(fullfile(data_folder, fn{i}));
    imglist = AppendPrefixPostfix(list, img_prefix, img_postfix);
    seglist  = AppendPrefixPostfix(list, seg_prefix, seg_postfix);

    assert(numel(imglist) == numel(seglist));
    
    fid = fopen(fullfile(save_folder, fn{i}), 'w');
    
    for j = 1 : numel(imglist)
        fprintf(fid, '%s %s\n', imglist{j}, seglist{j});
    end
    fclose(fid);
end



