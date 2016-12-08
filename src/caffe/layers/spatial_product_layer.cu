// Copyright Liang-Chieh Chen
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/spatial_product_layer.hpp"

namespace caffe {

template <typename Dtype>
void SpatialProductLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // Assume
  // bottom[0] dim NxCxHxW
  // bottom[1] dim Nx1xHxW
  // top[0]    dim NxCxHxW
  const int spatial_dim = bottom[0]->count(2);

  for (int n = 0; n < num_; ++n) {
      const Dtype* scale_data  = bottom[1]->gpu_data_at(n);
      for (int c = 0; c < channels_; ++c) {
        Dtype* top_data          = top[0]->mutable_gpu_data_at(n, c);
        const Dtype* bottom_data = bottom[0]->mutable_gpu_data_at(n, c);
        caffe_gpu_mul<Dtype>(spatial_dim, bottom_data, scale_data, top_data);
    }
  }
}

template <typename Dtype>
void SpatialProductLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const int count = bottom[0]->count();
  const int spatial_dim = bottom[0]->count(2);
  const int sample_dim = bottom[0]->count(1);
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    caffe_copy(count, top_diff, bottom_diff);
    for (int n = 0; n < num_; ++n) {
      const Dtype* scale_data = bottom[1]->gpu_data_at(n);
      for (int c = 0; c < channels_; ++c) {
        caffe_gpu_mul<Dtype>(spatial_dim, bottom_diff, scale_data, bottom_diff);
        bottom_diff += spatial_dim;
      }
    }
  }
  if (propagate_down[1]) {
    for (int n = 0; n < num_; ++n) {
      const Dtype* top_diff    = top[0]->gpu_diff_at(n);
      const Dtype* bottom_data = bottom[0]->gpu_data_at(n);
      Dtype* scale_diff        = bottom[1]->mutable_gpu_diff_at(n);
      caffe_gpu_mul<Dtype>(sample_dim, top_diff, bottom_data,
                           backward_buff_.mutable_gpu_diff());
      // accumulate over channels
      caffe_gpu_gemv<Dtype>(CblasTrans, channels_, spatial_dim, 1.,
                            backward_buff_.gpu_diff(), multiplier_.gpu_data(), 0,
                            scale_diff);
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SpatialProductLayer);
}  // namespace caffe
