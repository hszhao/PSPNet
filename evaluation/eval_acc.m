function eval_acc(data_name,data_root,eval_list,save_gray_folder,data_class,numClass)
addpath('evaluationCode');
addpath('visualizationCode');
load(data_class);
list = importdata(fullfile(data_root,eval_list));
pathPred = save_gray_folder;
pathAnno = data_root;

%% Evaluation
% initialize statistics
cnt=0;
area_intersection = double.empty;
area_union = double.empty;
pixel_accuracy = double.empty;
pixel_correct = double.empty;
pixel_labeled = double.empty;

% main loop
filesPred = dir(fullfile(pathPred, '*.png'));
for i = 1: numel(filesPred)
    % check file existence
    str = strsplit(list{i});
    strPred = strsplit(str{2},'/');
    strPred = strPred{end};
    if(strcmp(data_name,'cityscapes'))
        strPred = strrep(strPred,'gtFine_labelTrainIds','leftImg8bit');
    end
    filePred = fullfile(pathPred, strPred);
    fileAnno = fullfile(pathAnno, str{2});
    if ~exist(fileAnno, 'file')
        fprintf('Label file [%s] does not exist!\n', fileAnno); continue;
    end

    % read in prediction and label
    imPred = imread(filePred);
    imAnno = imread(fileAnno);
    imAnno = imAnno + 1;
    if(strcmp(data_name,'VOC2012') || strcmp(data_name,'cityscapes'))
    	imPred = imPred + 1;
    end
    imAnno(imAnno==255) = 0;
    imPred = imresize(imPred,[size(imAnno,1), size(imAnno,2)],'nearest');
    
    % check image size
    if size(imPred, 3) ~= 1
        fprintf('Label image [%s] should be a gray-scale image!\n', fileAnno); continue;
    end
    if size(imPred, 1)~=size(imAnno, 1) || size(imPred, 2)~=size(imAnno, 2)
        fprintf('Label image [%s] should have the same size as label image! Resizing...\n', fileLab);
        imPred = imresize(imPred, size(imAnno));
    end

    % compute IoU
    cnt = cnt + 1;
    [area_intersection(:,cnt), area_union(:,cnt)] = intersectionAndUnion(imPred, imAnno, numClass);

    % compute pixel-wise accuracy
    [pixel_accuracy(i), pixel_correct(i), pixel_labeled(i)] = pixelAccuracy(imPred, imAnno);
    fprintf('Evaluating %d/%d... Pixel-wise accuracy: %2.2f%%\n', cnt, numel(filesPred), pixel_accuracy(i)*100.);
end

%% Summary
IoU = sum(area_intersection,2)./sum(eps+area_union,2);
mean_IoU = mean(IoU);
accuracy = sum(pixel_correct)/sum(pixel_labeled);

fprintf('==== Summary IoU ====\n');
for i = 1:numClass
    fprintf('%3d %16s: %.4f\n', i, objectNames{i}, IoU(i));
end
fprintf('Mean IoU over %d classes: %.4f\n', numClass, mean_IoU);
fprintf('Pixel-wise Accuracy: %2.2f%%\n', accuracy*100.);
sub_acc = accuracy*100.;
sub_iou = mean_IoU;
