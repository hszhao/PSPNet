#ifdef USE_CUDNN
#include <algorithm>
#include <cfloat>
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

#if CUDNN_VERSION_MIN(5, 0, 0)

namespace caffe {

template <typename Dtype>
void CuDNNBNLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  BNLayer<Dtype>::LayerSetUp(bottom, top);
  save_mean_.ReshapeLike(*(this->blobs_[2]));
  save_inv_variance_.ReshapeLike(*(this->blobs_[3]));

  // Initialize CUDNN.
  CUDNN_CHECK(cudnnCreate(&handle_));
  cudnn::createTensor4dDesc<Dtype>(&bottom_desc_);
  cudnn::createTensor4dDesc<Dtype>(&top_desc_);
  cudnn::createTensor4dDesc<Dtype>(&bn_param_desc_);
  handles_setup_ = true;
  
  LOG(INFO)<<"using cuDNN BN engine";
}

template <typename Dtype>
void CuDNNBNLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Do not call BNLayer::Reshape function as some members are unnecessary
  this->num_ = bottom[0]->num();
  this->channels_ = bottom[0]->channels();
  this->height_ = bottom[0]->height();
  this->width_ = bottom[0]->width();

  top[0]->ReshapeLike(*(bottom[0]));

  // CUDNN tensors
  cudnn::setTensor4dDesc<Dtype>(&bottom_desc_, this->num_, this->channels_,
                                this->height_, this->width_);
  cudnn::setTensor4dDesc<Dtype>(&top_desc_, this->num_, this->channels_,
                                this->height_, this->width_);
  // Fix to the spatial mode
  CUDNN_CHECK(cudnnDeriveBNTensorDescriptor(bn_param_desc_,
      bottom_desc_, CUDNN_BATCHNORM_SPATIAL));

  if (this->frozen_){
    this->broadcast_buffer_.ReshapeLike(*(bottom[0]));
    this->spatial_statistic_.Reshape(this->num_, this->channels_, 1, 1);
    this->batch_statistic_.Reshape(1, this->channels_, 1, 1);

    this->spatial_sum_multiplier_.Reshape(1, 1, this->height_, this->width_);
    caffe_set(this->spatial_sum_multiplier_.count(), Dtype(1),
      this->spatial_sum_multiplier_.mutable_cpu_data());
    this->batch_sum_multiplier_.Reshape(this->num_, 1, 1, 1);
    caffe_set(this->batch_sum_multiplier_.count(), Dtype(1),
      this->batch_sum_multiplier_.mutable_cpu_data());

  }
}

template <typename Dtype>
CuDNNBNLayer<Dtype>::~CuDNNBNLayer() {
  // Check that handles have been setup before destroying.
  if (!handles_setup_) { return; }

  cudnnDestroyTensorDescriptor(bottom_desc_);
  cudnnDestroyTensorDescriptor(top_desc_);
  cudnnDestroyTensorDescriptor(bn_param_desc_);
  cudnnDestroy(handle_);
}

INSTANTIATE_CLASS(CuDNNBNLayer);

}  // namespace caffe
#endif
#endif
