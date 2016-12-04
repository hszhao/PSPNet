function [area_intersection, area_union] = intersectionAndUnion(imPred, imLab, numClass)
%% This function takes the prediction and label of a single image, returns intersection and union areas for each class
% To compute over many images do:
% for i = 1:Nimages
%  [area_intersection(:,i), area_union(:,i)]=intersectionAndUnion(imPred{i}, imLab{i});
% end
% IoU = sum(area_intersection,2)./sum(eps+area_union,2);

imPred = uint16(imPred(:));
imLab = uint16(imLab(:));

% Remove classes from unlabeled pixels in label image. 
% We should not penalize detections in unlabeled portions of the image.
imPred = imPred.*uint16(imLab>0);

% Compute area intersection
intersection = imPred.*uint16(imPred==imLab);
area_intersection = hist(intersection, 0:numClass);

% Compute area union
area_pred = hist(imPred, 0:numClass);
area_lab = hist(imLab, 0:numClass);
area_union = area_pred + area_lab - area_intersection;

% Remove unlabeled bin and convert to uint64
area_intersection = area_intersection(2:end);
area_union = area_union(2:end);