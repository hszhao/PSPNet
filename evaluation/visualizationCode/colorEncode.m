function rgbLab = colorEncode(grayLab, colorcode)
%% This function encodes label maps into rgb images for better visualization
% grayLab: [h, w]
% colorcode: [n, 3] where n is the number of classes
% rgbLab: [h, w, 3]

[h, w] = size(grayLab);
rgbLab = zeros(h, w, 3, 'uint8');

idx_unique = unique(grayLab)';
for idx = idx_unique
    if idx==0   continue;   end
    rgbLab = rgbLab + bsxfun(@times, uint8(grayLab==idx), reshape(colorcode(idx,:), [1,1,3]));
end