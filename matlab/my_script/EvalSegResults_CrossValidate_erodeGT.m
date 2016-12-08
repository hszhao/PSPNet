
SetupEnv;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% You do not need to chage values below
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if is_server
    if strcmp(dataset, 'voc12')
        VOC_root_folder = '/rmt/data/pascal/VOCdevkit';
    elseif strcmp(dataset, 'coco')
        VOC_root_folder = '/rmt/data/coco';
    else
        error('Wrong dataset!');
    end
else
    if strcmp(dataset, 'voc12')
        VOC_root_folder = '~/dataset/PASCAL/VOCdevkit';
    elseif strcmp(dataset, 'coco')
        VOC_root_folder = '~/dataset/coco';
    else
        error('Wrong dataset!');
    end
end

%
best_avacc = -1;
best_w = -1;
best_x_std = -1;
best_r_std = -1;

%fid = fopen(sprintf('cross_avgIOU_%s_%s_%sDownSample%d.txt', dataset, feature_type, testset, num_sample), 'a');
fid = fopen(sprintf('cross_avgIOU_%s_%s_%sDownSample%d.txt', dataset, feature_name, feature_type, testset, num_sample), 'a');

for w = range_bi_w       %0.5:0.5:6 %[1 5 10 15 20]
  bi_w = w;
  for x_std = range_bi_x_std   %1:12 %[10 20 30 40 50]
    bi_x_std = x_std;
    for r_std = range_bi_r_std  %5:5:10      %[10 20 30 40 50]
      bi_r_std = r_std;

      for pw = range_pos_w
        pos_w = pw;

        for p_x_std = range_pos_x_std
          pos_x_std = p_x_std;

          post_folder = sprintf('post_densecrf_W%d_XStd%d_RStd%d_PosW%d_PosXStd%d_numSample%d', bi_w, bi_x_std, bi_r_std, pos_w, pos_x_std, num_sample);

          %save_root_folder = fullfile('/rmt/work/deeplabel/exper', dataset, 'res', feature_type, post_folder);
          save_root_folder = fullfile('/rmt/work/deeplabel/exper', dataset, 'res', feature_name, feature_type, post_folder);


          if strcmp(dataset, 'voc12')
              seg_res_dir = [save_root_folder '/results/VOC2012/'];
              seg_root = fullfile(VOC_root_folder, 'VOC2012');
          elseif strcmp(dataset, 'coco')
              seg_res_dir = [save_root_folder '/results/COCO2014/'];
              seg_root = fullfile(VOC_root_folder, '');
          else
              error('Wrong dataset!')
          end

          save_result_folder = fullfile(seg_res_dir, 'Segmentation', [id '_' testset '_cls']);      

          if ~exist(save_result_folder, 'dir')
            mkdir(save_result_folder);
          end

          if strcmp(dataset, 'voc12')
            VOCopts = GetVOCopts(seg_root, seg_res_dir, trainset, testset, 'VOC2012');
          elseif strcmp(dataset, 'coco')
            VOCopts = GetVOCopts(seg_root, seg_res_dir, trainset, testset, '');
          end      

          % get iou score
          [accuracies, avacc, conf, rawcounts] = MyVOCevalseg(VOCopts, id);

          if best_avacc < avacc
            best_avacc = avacc;
            best_accuracies = accuracies;
            best_conf = conf;
            best_rawcounts = rawcounts;

            best_w = w;
            best_x_std = x_std;
            best_r_std = r_std;
            best_pos_w = pos_w;
            best_pos_x_std = pos_x_std;
          end
        
      fprintf(fid, 'w %2.2f, x_std %2.2f, r_std %2.2f, pos_w %2.2f, pos_x_std %2.2f, avacc %6.3f%%\n', bi_w, bi_x_std, bi_r_std, pos_w, pos_x_std, avacc);

        end
      end      
    end
  end
end

fprintf(1, 'Best avacc %6.3f%% occurs at w = %2.2f, x_std = %2.2f, r_std = %2.2f, pos_w %2.2f, pos_x_std %2.2f\n', best_avacc, best_w, best_x_std, best_r_std, best_pos_w, best_pos_x_std);

    

fclose(fid);
