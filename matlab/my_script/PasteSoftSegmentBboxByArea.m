function prob= PasteSoftSegmentBboxByArea(segmentation, classes, num_class, type, erode_scale,  soft_erosion, occluder_bias, debug)
% first sort the instances within the segmentation map by area size
% paste the instances (in the format of bounding box) according to the area
% size

if nargin < 4
    type = 'descend';
end

if nargin < 5
    erode_scale = 1;
end

if nargin < 6
    soft_erosion = 0;
end

if nargin < 7
    occluder_bias = 0;
end

if nargin < 8
    debug = 0;
end

if ~strcmp(type, 'ascend') && ~strcmp(type, 'descend')
    error('wrong type\n')
end

assert(erode_scale <= 1 && erode_scale > 0);

erode_scale_margin = 0.05;

%[~, ind] = SortSegmentationByArea(segmentation, type);
[~, ind] = SortSegmentationByBbox(segmentation, type);

result = 255 * ones(size(segmentation));

% handle background class first
cur_tmp = zeros(size(segmentation));

for i = 1 : numel(ind)
    class = classes(ind(i));
    
    if class == 0 || class == 255
        continue;
    end
    
    [row col] = find(segmentation == ind(i) - 1);
    
    if isempty(row)
        continue;
    end
    bbox_row_max = max(row);
    bbox_row_min  = min(row);
    bbox_col_max  = max(col);
    bbox_col_min   = min(col);
    
    cur_tmp(bbox_row_min:bbox_row_max, bbox_col_min:bbox_col_max) = 1;
        
end
result(cur_tmp == 0) = 0;

bbox_size = cell(numel(ind), 2);
% handle foreground classes
for i = 1 : numel(ind)
%     if ind(i) == 0
%         continue;
%     end
    
    class = classes(ind(i));
  
    if class == 0 || class == 255
        continue;
    end
    
    [row col] = find(segmentation == ind(i) - 1);
    
    bbox_row_max = max(row);
    bbox_row_min  = min(row);
    bbox_col_max  = max(col);
    bbox_col_min   = min(col);
    
    bbox_height = bbox_row_max - bbox_row_min + 1;
    bbox_width  = bbox_col_max - bbox_col_min + 1;
    bbox_area    = bbox_height * bbox_width;
    
    bbox_size{i, 1} = [bbox_row_min, bbox_row_max, bbox_col_min, bbox_col_max];
    
    if erode_scale == 1 || bbox_area < 121 || bbox_height < 10 || bbox_width < 10 % too small to erode
        result(bbox_row_min:bbox_row_max, bbox_col_min:bbox_col_max) = class;
        bbox_size{i, 2} = [bbox_row_min, bbox_row_max, bbox_col_min, bbox_col_max];
    else
        % do erosion
        mask = zeros(size(result));
        
        mask(bbox_row_min:bbox_row_max, bbox_col_min:bbox_col_max) = 1;
        
        % make boundary to be zero so that the eroded mask will be centered
        mask(1,:) = 0;
        mask(end,:)=0;
        mask(:,1) = 0;
        mask(:,end)=0;
        
        %se = strel('square', floor(min(bbox_height, bbox_width) * erode_scale));
        %mask_erode = imerode(mask, se);
        se = strel('square', 3);
        
        mask_erode = imerode(mask, se);
        while (1)
            prop = sum(mask_erode(:)) / sum(mask(:));
            
            % add a small margin
            if prop < erode_scale + erode_scale_margin 
                break;
            end
            mask_erode = imerode(mask_erode, se);
        end
        
        if debug
            fprintf(1, '%f\n', sum(mask_erode(:)) / sum(mask(:)));
        end

        [row col] = find(mask_erode == 1);
        bbox_row_max = max(row);
        bbox_row_min  = min(row);
        bbox_col_max  = max(col);
        bbox_col_min   = min(col);
        bbox_size{i, 2} = [bbox_row_min, bbox_row_max, bbox_col_min, bbox_col_max];
        
        result(mask_erode == 1) = class;
    end
end

% transform the result to prob
zero_prob       = 1e-12;
one_prob        = 1 - zero_prob;

occluder_count = 0;   % 1st instance is not biased

prob = zero_prob * ones(size(result,1), size(result,2), num_class);

class_count = zeros(size(result, 1), size(result, 2), num_class);

for i = 1 : numel(ind)
    class = classes(ind(i));
  
    if class == 255
        continue;
    elseif class == 0
        pos = result == 0;
        tmp = zero_prob * ones(size(prob,1), size(prob,2));
        tmp(pos) = one_prob;        
        prob(:,:,class+1) = prob(:,:,class+1) + tmp;
        
        tmp = zeros(size(prob, 1), size(prob, 2));
        tmp(pos) = 1;
        class_count(:, :, 1) = class_count(:,:,1) + tmp;
    else
        orig_bbox = bbox_size{i,1};
        erod_bbox = bbox_size{i,2};
        
        tmp = TransformSegToProb(orig_bbox, erod_bbox, soft_erosion);    
        
        prob(orig_bbox(1):orig_bbox(2), orig_bbox(3):orig_bbox(4),class+1) = ...
            prob(orig_bbox(1):orig_bbox(2), orig_bbox(3):orig_bbox(4),class+1) + ...
            tmp;
        
        prob(orig_bbox(1):orig_bbox(2), orig_bbox(3):orig_bbox(4),1) = ...
            prob(orig_bbox(1):orig_bbox(2), orig_bbox(3):orig_bbox(4),1) + (1 - tmp);
        
        tmp = zeros(size(prob, 1), size(prob, 2));
        tmp(orig_bbox(1):orig_bbox(2), orig_bbox(3):orig_bbox(4)) = 1;
        class_count(:,:, class+1) = class_count(:,:, class+1) + tmp;
        class_count(:,:, 1)           = class_count(:,:, 1) + tmp;
        
        total_count = sum(class_count(erod_bbox(1):erod_bbox(2), erod_bbox(3):erod_bbox(4)), 3);
        overlap_pos = find(total_count > 1);  % it is overlapped
        if ~isempty(overlap_pos)
            % smaller objects have higher weight
            occluder_count = occluder_count + 1;
        
            if occluder_count > 0
                overlap_mask = zeros(erod_bbox(2)-erod_bbox(1)+1, erod_bbox(4)-erod_bbox(3)+1);
                overlap_mask(overlap_pos) = occluder_count * occluder_bias;
                prob(erod_bbox(1):erod_bbox(2), erod_bbox(3):erod_bbox(4), class+1) = ...
                    prob(erod_bbox(1):erod_bbox(2), erod_bbox(3):erod_bbox(4), class+1) + overlap_mask;
            end
        end
        
        %class_count(orig_bbox(1):orig_bbox(2), orig_bbox(3):orig_bbox(4)) = ...
         %   class_count(orig_bbox(1):orig_bbox(2), orig_bbox(3):orig_bbox(4)) + 1;
    end
 
    
end

% normalize again
for i = 1 : size(class_count, 3)
    tmp = prob(:, :, i);
    tmp2 = class_count(:, :, i);
    
    ind = tmp2 ~= 0;
    
    tmp(ind) = tmp(ind) ./ tmp2(ind);
    prob(:, :, i) = tmp;
end

prob = bsxfun(@rdivide, prob, sum(prob, 3));
