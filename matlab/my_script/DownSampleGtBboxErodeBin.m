% downsample the bin files for faster cross-validation and not overfit val set
% 
addpath('/rmt/work/deeplabel/code/matlab/my_script');
SetupEnv;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% You do not need to chage values below
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if is_server
  %save_folder  = fullfile('/rmt/work/deeplabel/exper', dataset, feature_type, 'bin');
  save_folder  = fullfile('/rmt/work/deeplabel/exper', dataset, feature_name, feature_type);
else
  save_folder = '../feature_bin';
end

dest_folder = [save_folder, sprintf('_numSample%d', num_sample)];

if ~exist(dest_folder, 'dir')
  mkdir(dest_folder)
end

% get testset list
if strcmp(testset, 'test')
  error('Cannot downsample for testset\n')
else
  if strcmp(dataset, 'voc12')
    VOC_root_folder = '/rmt/data/pascal/VOCdevkit/VOC2012';
  elseif strcmp(dataset, 'coco')
    VOC_root_folder = '/rmt/data/coco';
  else
    error('Wrong dataset!');
  end
  fid = fopen(fullfile(VOC_root_folder, 'ImageSets/Segmentation', [testset '.txt']), 'r');
  gtids = textscan(fid, '%s');
  gtids = gtids{1};
  fclose(fid);
end

if down_sample_method == 1
  gtids = gtids(1:down_sample_rate:end);
elseif down_sample_method == 2
  ind = randperm(length(gtids));
  ind = ind(1:num_sample);
  gtids = gtids(ind);
else
  error('Wrong down_sample_method\n');
end

for i = 1 : length(gtids)
  fprintf(1, 'processing %d (%d)...\n', i, length(gtids));
  copyfile(fullfile(save_folder, [gtids{i} '.bin']), fullfile(dest_folder, [gtids{i} '.bin']));
end
