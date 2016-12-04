function data_output = scale_process(net,img_scale,fea_cha,crop_size,ori_rows,ori_cols,mean_r,mean_g,mean_b)
    data_output = zeros(ori_rows,ori_cols,fea_cha,'single');
    new_rows = size(img_scale,1);
    new_cols = size(img_scale,2);
    long_size = new_rows;
    short_size = new_cols;
    if(new_cols > long_size)
       long_size = new_cols;
       short_size = new_rows;
    end
    if(long_size <= crop_size)
        input_data = pre_img(img_scale,crop_size,mean_r,mean_g,mean_b);
        score = caffe_process(input_data,net);
        score = score(1:new_rows,1:new_cols,:);
    else
        stride_rate = 2/3;
        stride = ceil(crop_size*stride_rate);
        img_pad = img_scale;
        if(short_size < crop_size)
          if(new_rows < crop_size)
              im_r = padarray(img_pad(:,:,1),[crop_size-new_rows,0],mean_r,'post');
              im_g = padarray(img_pad(:,:,2),[crop_size-new_rows,0],mean_g,'post');
              im_b = padarray(img_pad(:,:,3),[crop_size-new_rows,0],mean_b,'post');
              img_pad = cat(3,im_r,im_g,im_b);
          end
          if(new_cols < crop_size)
              im_r = padarray(img_pad(:,:,1),[0,crop_size-new_cols],mean_r,'post');
              im_g = padarray(img_pad(:,:,2),[0,crop_size-new_cols],mean_g,'post');
              im_b = padarray(img_pad(:,:,3),[0,crop_size-new_cols],mean_b,'post');
              img_pad = cat(3,im_r,im_g,im_b);
          end
        end
        pad_rows = size(img_pad,1);
        pad_cols = size(img_pad,2);
        h_grid = ceil(single(pad_rows-crop_size)/stride) + 1;
        w_grid = ceil(single(pad_cols-crop_size)/stride) + 1;
        data_scale = zeros(pad_rows,pad_cols,fea_cha,'single');
        count_scale = zeros(pad_rows,pad_cols,fea_cha,'single');
        for grid_yidx=1:h_grid
            for grid_xidx=1:w_grid
                s_x = (grid_xidx-1) * stride + 1;
                s_y = (grid_yidx-1) * stride + 1;
                e_x = min(s_x + crop_size - 1, pad_cols);
                e_y = min(s_y + crop_size - 1, pad_rows);
                s_x = e_x - crop_size + 1;
                s_y = e_y - crop_size + 1;
                img_sub = img_pad(s_y:e_y,s_x:e_x,:);
                count_scale(s_y:e_y,s_x:e_x,:) = count_scale(s_y:e_y,s_x:e_x,:) + 1;
                input_data = pre_img(img_sub,crop_size,mean_r,mean_g,mean_b);
                data_scale(s_y:e_y,s_x:e_x,:) = data_scale(s_y:e_y,s_x:e_x,:) + caffe_process(input_data,net);
            end
        end
        score = data_scale./count_scale;
        score = score(1:new_rows,1:new_cols,:); 
    end

    data_output = imresize(score,[ori_rows ori_cols],'bilinear');
    data_output = bsxfun(@rdivide, data_output, sum(data_output, 3));
end