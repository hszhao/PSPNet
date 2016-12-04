function eval_sub(data_name,data_root,eval_list,model_weights,model_deploy,fea_cha,base_size,crop_size,data_class,data_colormap, ...
		  is_save_feat,save_gray_folder,save_color_folder,save_feat_folder,gpu_id,index,step,skipsize,scale_array,mean_r,mean_g,mean_b)
list = importdata(fullfile(data_root,eval_list));
load(data_class);
load(data_colormap);
if(~isdir(save_gray_folder))
    mkdir(save_gray_folder);
end
if(~isdir(save_color_folder))
    mkdir(save_color_folder);
end
if(~isdir(save_feat_folder) && is_save_feat)
    mkdir(save_feat_folder);
end

phase = 'test'; %run with phase test (so that dropout isn't applied)
if ~exist(model_weights, 'file')
  error('Model missing!');
end
caffe.reset_all();
caffe.set_mode_gpu();
caffe.set_device(gpu_id);
net = caffe.Net(model_deploy, model_weights, phase);

for i = skipsize+(index-1)*step+1:skipsize+index*step
    fprintf(1, 'processing %d (%d)...\n', i, numel(list));
    str = strsplit(list{i});
    img = imread(fullfile(data_root,str{1}));
    if(size(img,3) < 3) %for gray image
    	im_r = img;
    	im_g = img;
    	im_b = img;
	img = cat(3,im_r,im_g,im_b);
    end
    ori_rows = size(img,1);
    ori_cols = size(img,2);
    data_all = zeros(ori_rows,ori_cols,fea_cha,'single');
    for j = 1:size(scale_array,2)
        long_size = base_size*scale_array(j) + 1;
        new_rows = long_size;
        new_cols = long_size;
        if ori_rows > ori_cols
            new_cols = round(long_size/single(ori_rows)*ori_cols);
        else
            new_rows = round(long_size/single(ori_cols)*ori_rows);
        end 
        img_scale = imresize(img,[new_rows new_cols],'bilinear');
        data_all = data_all + scale_process(net,img_scale,fea_cha,crop_size,ori_rows,ori_cols,mean_r,mean_g,mean_b);
    end
    
    data_all = data_all/size(scale_array,2);
    data = data_all; %already exp process

    img_fn = strsplit(str{1},'/');
    img_fn = img_fn{end};
    img_fn = img_fn(1:end-4);

    [~,imPred] = max(data,[],3);
    imPred = uint8(imPred);
    
    switch data_name
        case 'ADE20K'
            rgbPred = colorEncode(imPred, colors);
            imwrite(imPred,[save_gray_folder img_fn '.png']);
            imwrite(rgbPred,[save_color_folder img_fn '.png']);
        case 'VOC2012'
            imPred = imPred - 1;
            imwrite(imPred,[save_gray_folder img_fn '.png']);
            imwrite(imPred,colormapvoc,[save_color_folder img_fn '.png']);
        case 'cityscapes'
            imPred = imPred - 1;
            imwrite(imPred,[save_gray_folder img_fn '.png']);
            imwrite(imPred,colormapcs,[save_color_folder img_fn '.png']);
    end

    if(is_save_feat)
        save([save_feat_folder img_fn],'data');
    end
end
caffe.reset_all();
end
