function [pixel_accuracy, pixel_correct, pixel_labeled]= pixelAccuracy(imPred, imLab)
%% This function takes the prediction and label of a single image, returns pixel-wise accuracy
% To compute over many images do:
% for i = 1:Nimages
%  [pixel_accuracy(i), pixel_correct(i), pixel_labeled(i)] = pixelAccuracy(imPred{i}, imLab{i});
% end
% mean_pixel_accuracy = sum(pixel_correct)/(eps + sum(pixel_labeled));

imPred = uint16(imPred(:));
imLab = uint16(imLab(:));

% Remove classes from unlabeled pixels in gt image. 
% We should not penalize detections in unlabeled portions of the image.
pixel_labeled = sum(imLab>0);
pixel_correct = sum((imPred==imLab).*(imLab>0));
pixel_accuracy = pixel_correct / pixel_labeled;
