function [val ind] = SortSegmentationByBbox(segmentation, type)
% sort instances within the segmentation map by area size
%

if nargin < 2
    type = 'descend';
end

if ~strcmp(type, 'ascend') && ~strcmp(type, 'descend')
    error('wrong type\n')
end

labels = unique(segmentation(:));

areas = zeros(1, numel(labels));

for i = 1 : numel(labels)
    [row col] = find(segmentation == labels(i));
    
    areas(i) = (max(row) - min(row)) * (max(col) - min(col));
end

[val ind] = sort(areas, type);

%ind = ind - 1;  %instance id starts from 0
