% perform net surgery on ResNet-101
% change all the 2048 filters to 1024
%

addpath ./matlab

save_weights_name = 'ResNet-101-model-1024.caffemodel';

root_folder = '/rmt/work/deeplabel/exper/msra';

model = fullfile(root_folder, 'ResNet-101-deploy.prototxt');
weights = fullfile(root_folder, 'ResNet-101-model.caffemodel');

caffe.set_mode_gpu();
caffe.set_device(0);

net = caffe.Net(model, weights, 'test');

layer_names = {...
	       '5a_branch1',...
	       '5a_branch2c',...
	       '5b_branch2c',...
	       '5c_branch2c',...
};

for ii = 1 : numel(layer_names)
  layer_name = layer_names{ii};
  tmp_param = net.params(['res' layer_name], 1).get_data();
  tmp_param = tmp_param(:,:,:,1:2:end);
  net.params(['res' layer_name], 1).reshape(size(tmp_param));
  net.params(['res' layer_name], 1).set_data(tmp_param);

  for i = 1 : 2
    tmp_param = net.params(['bn' layer_name], i).get_data();
    tmp_param = tmp_param(1:2:end);
    net.params(['bn' layer_name], i).reshape(length(tmp_param));
    net.params(['bn' layer_name], i).set_data(tmp_param);
  end

  for i = 1 : 2
    tmp_param = net.params(['scale' layer_name], i).get_data();
    tmp_param = tmp_param(1:2:end);
    net.params(['scale' layer_name], i).reshape(length(tmp_param));
    net.params(['scale' layer_name], i).set_data(tmp_param);
  end
end

layer_names = {...
	       '5b_branch2a',...
	       '5c_branch2a',...
};

for ii = 1 : numel(layer_names)
  layer_name = layer_names{ii};
  tmp_param = net.params(['res' layer_name], 1).get_data();
  tmp_param = tmp_param(:,:,1:2:end,:);
  net.params(['res' layer_name], 1).reshape(size(tmp_param));
  net.params(['res' layer_name], 1).set_data(tmp_param);
end

net.save(fullfile(root_folder, save_weights_name));

caffe.reset_all();
