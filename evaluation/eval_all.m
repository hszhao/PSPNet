%{
Variables need to be modified: data_root, eval_list;
and the default GPUs used for evaluation are with ID [0:3],
modify variable 'gpu_id_array' if needed.
%}

close all; clc; clear;
addpath('../matlab'); %add matcaffe path
addpath('visualizationCode');
data_name = 'ADE20K'; %set to 'VOC2012' or 'cityscapes' for relevant datasets

switch data_name
    case 'ADE20K'
        isVal = true; %evaluation on valset
        step = 500; %equals to number of images divide num of GPUs in testing e.g. 500=2000/4
        data_root = '/data2/hszhao/dataset/ADEChallengeData2016'; %root path of dataset
        eval_list = 'list/ADE20K_val.txt'; %evaluation list, refer to lists in folder 'samplelist'
        save_root = 'mc_result/ADE20K/val/pspnet50_473/'; %root path to store the result image
        model_weights = 'model/pspnet50_ADE20K.caffemodel';
        model_deploy = 'prototxt/pspnet50_ADE20K_473.prototxt';
        fea_cha = 150; %number of classes
        base_size = 512; %based size for scaling
        crop_size = 473; %crop size fed into network
        data_class = 'objectName150.mat'; %class name
        data_colormap = 'color150.mat'; %color map
    case 'VOC2012'
        isVal = false; %evaluation on testset
        step = 364; %364=1456/4
        data_root = '/data2/hszhao/dataset/VOC2012';
        eval_list = 'list/VOC2012_test.txt';
        save_root = 'mc_result/VOC2012/test/pspnet101_473/';
        model_weights = 'model/pspnet101_VOC2012.caffemodel';
        model_deploy = 'prototxt/pspnet101_VOC2012_473.prototxt';
        fea_cha = 21;
        base_size = 512;
        crop_size = 473;
        data_class = 'objectName21.mat';
        data_colormap = 'colormapvoc.mat';
    case 'cityscapes'
        isVal = true;
        step = 125; %125=500/4
        data_root = '/data2/hszhao/dataset/cityscapes';
        eval_list = 'list/cityscapes_val.txt';
        save_root = 'mc_result/cityscapes/val/pspnet101_713/';
        model_weights = 'model/pspnet101_cityscapes.caffemodel';
        model_deploy = 'prototxt/pspnet101_cityscapes_713.prototxt';
        fea_cha = 19;
        base_size = 2048;
        crop_size = 713;
        data_class = 'objectName19.mat';
        data_colormap = 'colormapcs.mat';
end
skipsize = 0; %skip serveal images in the list

is_save_feat = false; %set to true if final feature map is needed (not suggested for storage consuming)
save_gray_folder = [save_root 'gray/']; %path for predicted gray image
save_color_folder = [save_root 'color/']; %path for predicted color image
save_feat_folder = [save_root 'feat/']; %path for predicted feature map
scale_array = [1]; %set to [0.5 0.75 1 1.25 1.5 1.75] for multi-scale testing
mean_r = 123.68; %means to be subtracted and the given values are used in our training stage
mean_g = 116.779;
mean_b = 103.939;

acc = double.empty;
iou = double.empty;
gpu_id_array = [0:3]; %multi-GPUs for parfor testing, if number of GPUs is changed, remember to change the variable 'step'
runID = 1;
gpu_num = size(gpu_id_array,2);
index_array = [(runID-1)*gpu_num+1:runID*gpu_num];

parfor i = 1:gpu_num %change 'parfor' to 'for' if singe GPU testing is used
  eval_sub(data_name,data_root,eval_list,model_weights,model_deploy,fea_cha,base_size,crop_size,data_class,data_colormap, ...
           is_save_feat,save_gray_folder,save_color_folder,save_feat_folder,gpu_id_array(i),index_array(i),step,skipsize,scale_array,mean_r,mean_g,mean_b);
end
if(isVal)
   eval_acc(data_name,data_root,eval_list,save_gray_folder,data_class,fea_cha);
end
