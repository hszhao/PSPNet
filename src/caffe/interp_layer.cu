#include <algorithm>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {


template <typename Dtype>
void InterpLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    NOT_IMPLEMENTED;
}

template <typename Dtype>
void InterpLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
}

INSTANTIATE_LAYER_GPU_FUNCS(InterpLayer);


}  // namespace caffe
