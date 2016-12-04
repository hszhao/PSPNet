function score = ms_caffe_process(input_data,net)
    score = net.forward({input_data});
    score = score{1};
    score_flip = net.forward({input_data(end:-1:1,:,:)});
    score_flip = score_flip{1};
    score = score + score_flip(end:-1:1,:,:);

    score = permute(score, [2 1 3]);
    score = exp(score);
    score = bsxfun(@rdivide, score, sum(score, 3));
end