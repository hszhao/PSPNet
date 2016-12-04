function colormap = colorMap(imPred, imAnno, objectnames)
%% This function generates colormaps with text annotations for visualization
% imPred, imAnno: [h, w]
% colormap: [h, w, 3]

colormap = cell(2,8);

idxUnique = unique([imPred, imAnno])';
cnt = 0;
for idx = idxUnique
    if idx==0   
        continue;   
    else
        cnt = cnt + 1;
        colormap{cnt} = imread([objectnames{idx} '.jpg']);
    end
    
    if cnt>= numel(colormap)
        break; 
    end
end

for i = cnt+1 : numel(colormap)
    colormap{i} = 255 * ones(30, 150, 3, 'uint8');
end

colormap = cell2mat(colormap');