// Copyright Liang-Chieh Chen
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/spatial_product_layer.hpp"

namespace caffe {

template <typename Dtype>
void SpatialProductLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_GE(bottom[0]->num_axes(), 2)
      << "Number of axes of bottom blob must be >=2.";
  CHECK_EQ(bottom.size(), 2)
      << "Current implementatin assumes only two bottom data.";
  CHECK_EQ(bottom[1]->channels(), 1)
      << "Current implementation assumes second input has only one channel.";
  CHECK_EQ(bottom[0]->width(), bottom[1]->width())
      << "bottom[0] and bottom[1] should have the same width.";
  CHECK_EQ(bottom[0]->height(), bottom[1]->height())
      << "bottom[0] and bottom[1] should have the same height.";
  
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_   = bottom[0]->height();
  width_    = bottom[0]->width();

  multiplier_.Reshape(1, channels_, 1, 1);
  backward_buff_.Reshape(1, channels_, height_, width_);
  caffe_set(multiplier_.count(), Dtype(1), multiplier_.mutable_cpu_data());
}

template <typename Dtype>
void SpatialProductLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_GE(bottom[0]->num_axes(), 2)
      << "Number of axes of bottom blob must be >=2.";
  top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void SpatialProductLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // Assume
  // bottom[0] dim NxCxHxW
  // bottom[1] dim Nx1xHxW
  // top[0]    dim NxCxHxW
  const int spatial_dim = bottom[0]->count(2);

  for (int n = 0; n < num_; ++n) {
      const Dtype* scale_data  = bottom[1]->cpu_data_at(n);
      for (int c = 0; c < channels_; ++c) {
        Dtype* top_data          = top[0]->mutable_cpu_data_at(n, c);
        const Dtype* bottom_data = bottom[0]->mutable_cpu_data_at(n, c);
        caffe_mul<Dtype>(spatial_dim, bottom_data, scale_data, top_data);
    }
  }
}

template <typename Dtype>
void SpatialProductLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const int count = bottom[0]->count();
  const int spatial_dim = bottom[0]->count(2);
  const int sample_dim = bottom[0]->count(1);
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_copy(count, top_diff, bottom_diff);
    for (int n = 0; n < num_; ++n) {
      const Dtype* scale_data = bottom[1]->cpu_data_at(n);
      for (int c = 0; c < channels_; ++c) {
        caffe_mul<Dtype>(spatial_dim, bottom_diff, scale_data, bottom_diff);
        bottom_diff += spatial_dim;
      }
    }
  }
  if (propagate_down[1]) {
    for (int n = 0; n < num_; ++n) {
      const Dtype* top_diff    = top[0]->cpu_diff_at(n);
      const Dtype* bottom_data = bottom[0]->cpu_data_at(n);
      Dtype* scale_diff        = bottom[1]->mutable_cpu_diff_at(n);
      caffe_mul<Dtype>(sample_dim, top_diff, bottom_data,
                       backward_buff_.mutable_cpu_diff());
      // accumulate over channels
      caffe_cpu_gemv<Dtype>(CblasTrans, channels_, spatial_dim, 1.,
                 backward_buff_.cpu_diff(), multiplier_.cpu_data(), 0,
                 scale_diff);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(SpatialProductLayer);
#endif

INSTANTIATE_CLASS(SpatialProductLayer);
REGISTER_LAYER_CLASS(SpatialProduct);

}  // namespace caffe
