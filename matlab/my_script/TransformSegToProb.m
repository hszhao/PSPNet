function frg_prob = TransformSegToProb(orig_bbox_size, erod_bbox_size, soft_erosion)
% transform the ground truth annotations to prob, used for analysis
% experiments
% soft_erosion: interpolate the values between
%

zero_prob       = 1e-12;
%one_prob        = 1 - zero_prob;
one_prob        = 0.75 - zero_prob;    % weak one probability
uniform_prob = 1 / 2;

orig_bbox_num_row = orig_bbox_size(2) - orig_bbox_size(1) + 1;
orig_bbox_num_col  = orig_bbox_size(4) - orig_bbox_size(3) + 1;

erod_bbox_min_row = erod_bbox_size(1) - orig_bbox_size(1) + 1;
erod_bbox_max_row = erod_bbox_size(2) - orig_bbox_size(1) + 1;
erod_bbox_min_col = erod_bbox_size(3) - orig_bbox_size(3) + 1;
erod_bbox_max_col = erod_bbox_size(4) - orig_bbox_size(3) + 1;

frg_prob = zero_prob * ones(orig_bbox_num_row, orig_bbox_num_col);
if soft_erosion
    mask = zeros(erod_bbox_max_row-erod_bbox_min_row+3, erod_bbox_max_col-erod_bbox_min_col+3);
    mask(2:end-1, 2:end-1) = 1;
    
    [x y] = meshgrid(1:size(mask,2), 1:size(mask,1));
    x_grid = unique([linspace(1,2, erod_bbox_size(3) - orig_bbox_size(3)+2), 1:size(mask,2), ...
        linspace(size(mask,2)-1,size(mask,2), orig_bbox_size(4) - erod_bbox_size(4) +2)]);
    y_grid = unique([linspace(1,2, erod_bbox_size(1) - orig_bbox_size(1)+2), 1:size(mask,1), ...
        linspace(size(mask,1)-1,size(mask,1), orig_bbox_size(2) - erod_bbox_size(2)+2)]);
    [xi yi] = meshgrid(x_grid, y_grid);
    
    tmp = interp2(x, y, mask, xi, yi);
    frg_prob = tmp(2:end-1, 2:end-1);
else
    mask = false(size(frg_prob));
    mask(erod_bbox_min_row:erod_bbox_max_row, erod_bbox_min_col:erod_bbox_max_col) = 1;
    
    frg_prob(mask) = one_prob;
    frg_prob(~mask) = uniform_prob;
end


