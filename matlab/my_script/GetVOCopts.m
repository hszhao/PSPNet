function VOCopts = GetVOCopts(seg_root, seg_res_dir, trainset, testset, dataset)
%clear VOCopts

if nargin < 5
    dataset = 'VOC2012';
end

% dataset
%
% Note for experienced users: the VOC2008-11 test sets are subsets
% of the VOC2012 test set. You don't need to do anything special
% to submit results for VOC2008-11.

VOCopts.dataset=dataset;

% get devkit directory with forward slashes
devkitroot=strrep(fileparts(fileparts(mfilename('fullpath'))),'\','/');

% change this path to point to your copy of the PASCAL VOC data
VOCopts.datadir=[devkitroot '/'];

% change this path to a writable directory for your results
%VOCopts.resdir=[devkitroot '/results/' VOCopts.dataset '/'];
VOCopts.resdir = seg_res_dir;

% change this path to a writable local directory for the example code
VOCopts.localdir=[devkitroot '/local/' VOCopts.dataset '/'];

% initialize the training set

VOCopts.trainset = trainset;
%VOCopts.trainset='train'; % use train for development
% VOCopts.trainset='trainval'; % use train+val for final challenge

% initialize the test set

VOCopts.testset = testset;
%VOCopts.testset='val'; % use validation data for development test set
% VOCopts.testset='test'; % use test set for final challenge

% initialize main challenge paths

%VOCopts.annopath=[VOCopts.datadir VOCopts.dataset '/Annotations/%s.xml'];
%VOCopts.imgpath=[VOCopts.datadir VOCopts.dataset '/JPEGImages/%s.jpg'];
%VOCopts.imgsetpath=[VOCopts.datadir VOCopts.dataset '/ImageSets/Main/%s.txt'];
%VOCopts.clsimgsetpath=[VOCopts.datadir VOCopts.dataset '/ImageSets/Main/%s_%s.txt'];

VOCopts.annopath=[seg_root '/Annotations/%s.xml'];
VOCopts.imgpath=[seg_root '/JPEGImages/%s.jpg'];
VOCopts.imgsetpath=[seg_root '/ImageSets/Main/%s.txt'];
VOCopts.clsimgsetpath=[seg_root '/ImageSets/Main/%s_%s.txt'];


VOCopts.clsrespath=[VOCopts.resdir 'Main/%s_cls_' VOCopts.testset '_%s.txt'];
VOCopts.detrespath=[VOCopts.resdir 'Main/%s_det_' VOCopts.testset '_%s.txt'];

% initialize segmentation task paths

%if strcmp(dataset, 'VOC2012')
 %   VOCopts.seg.clsimgpath=[seg_root '/SegmentationClassAug/%s.png'];
%else
    VOCopts.seg.clsimgpath=[seg_root '/SegmentationClass/%s.png'];
%end

VOCopts.seg.instimgpath=[seg_root '/SegmentationObject/%s.png'];
VOCopts.seg.imgsetpath=[seg_root '/ImageSets/Segmentation/%s.txt'];

%VOCopts.seg.clsimgpath=[VOCopts.datadir VOCopts.dataset '/SegmentationClass/%s.png'];
%VOCopts.seg.instimgpath=[VOCopts.datadir VOCopts.dataset '/SegmentationObject/%s.png'];
%VOCopts.seg.imgsetpath=[VOCopts.dataset '/ImageSets/Segmentation/%s.txt'];


VOCopts.seg.clsresdir=[VOCopts.resdir 'Segmentation/%s_%s_cls'];
VOCopts.seg.instresdir=[VOCopts.resdir 'Segmentation/%s_%s_inst'];
VOCopts.seg.clsrespath=[VOCopts.seg.clsresdir '/%s.png'];
VOCopts.seg.instrespath=[VOCopts.seg.instresdir '/%s.png'];

% initialize layout task paths

VOCopts.layout.imgsetpath=[VOCopts.datadir VOCopts.dataset '/ImageSets/Layout/%s.txt'];
VOCopts.layout.respath=[VOCopts.resdir 'Layout/%s_layout_' VOCopts.testset '.xml'];

% initialize action task paths

VOCopts.action.imgsetpath=[VOCopts.datadir VOCopts.dataset '/ImageSets/Action/%s.txt'];
VOCopts.action.clsimgsetpath=[VOCopts.datadir VOCopts.dataset '/ImageSets/Action/%s_%s.txt'];
VOCopts.action.respath=[VOCopts.resdir 'Action/%s_action_' VOCopts.testset '_%s.txt'];

% initialize the VOC challenge options

% classes

if ~isempty(strfind(seg_root, 'VOC'))
  VOCopts.classes={...
    'aeroplane'
    'bicycle'
    'bird'
    'boat'
    'bottle'
    'bus'
    'car'
    'cat'
    'chair'
    'cow'
    'diningtable'
    'dog'
    'horse'
    'motorbike'
    'person'
    'pottedplant'
    'sheep'
    'sofa'
    'train'
    'tvmonitor'};
  
elseif ~isempty(strfind(seg_root, 'coco')) || ~isempty(strfind(seg_root, 'COCO'))
  coco_categories = GetCocoCategories();
  VOCopts.classes = coco_categories.values();
else
  error('Unknown dataset!\n');
end
 
VOCopts.nclasses=length(VOCopts.classes);	


% poses

VOCopts.poses={...
    'Unspecified'
    'Left'
    'Right'
    'Frontal'
    'Rear'};

VOCopts.nposes=length(VOCopts.poses);

% layout parts

VOCopts.parts={...
    'head'
    'hand'
    'foot'};    

VOCopts.nparts=length(VOCopts.parts);

VOCopts.maxparts=[1 2 2];   % max of each of above parts

% actions

VOCopts.actions={...    
    'other'             % skip this when training classifiers
    'jumping'
    'phoning'
    'playinginstrument'
    'reading'
    'ridingbike'
    'ridinghorse'
    'running'
    'takingphoto'
    'usingcomputer'
    'walking'};

VOCopts.nactions=length(VOCopts.actions);

% overlap threshold

VOCopts.minoverlap=0.5;

% annotation cache for evaluation

VOCopts.annocachepath=[VOCopts.localdir '%s_anno.mat'];

% options for example implementations

VOCopts.exfdpath=[VOCopts.localdir '%s_fd.mat'];
