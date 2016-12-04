#include <algorithm>
#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

#define THREAD_BLOCK_SIZE 256

template <typename Dtype>
__global__ void mean_statistic(const int num, const int map_size, const int channels, 
    Dtype stat_ratio, bool save_mean, bool moving_mean, Dtype decay, Dtype com_decay,
    const Dtype* in, Dtype* mean, Dtype* history_mean, Dtype* out, int norm_size) {
  __shared__ Dtype buffer[THREAD_BLOCK_SIZE]; 
  buffer[threadIdx.x] = 0;
  if(!moving_mean) {
    for(int i = threadIdx.x; i < num * map_size; i += blockDim.x) {
      int location = i / map_size * map_size * channels + (i % map_size) + blockIdx.x * map_size;
      if(i < num * map_size)
        buffer[threadIdx.x] += in[location];
    }
    __syncthreads();
    for(int i = blockDim.x / 2; i > 0; i >>= 1) {
      if(threadIdx.x < i) buffer[threadIdx.x] += buffer[threadIdx.x + i];
      __syncthreads();
    }
    if(threadIdx.x == 0) {
      buffer[0] = buffer[0] * stat_ratio;
      if(save_mean) mean[blockIdx.x] += (decay * buffer[0] + com_decay * history_mean[blockIdx.x]) / norm_size;
    }
  }
  else if(threadIdx.x == 0)
    buffer[0] = history_mean[blockIdx.x];

  __syncthreads();

  for(int i = threadIdx.x; i < num * map_size; i += blockDim.x) {
    int location = i / map_size * map_size * channels + (i % map_size) + blockIdx.x * map_size;
    if(i < num * map_size)
      out[location] = in[location] - buffer[0];
  }
}

template <typename Dtype>
__global__ void var_statistic(const int num, const int map_size, const int channels, 
    Dtype in_pow, Dtype stat_ratio, Dtype stat_eps, Dtype stat_pow,
    bool save_mean, bool moving_mean, Dtype decay, Dtype com_decay,
    const Dtype* in, Dtype* mean, Dtype* history_mean, Dtype* out,
    Dtype* x_norm,Dtype* x_std, const Dtype* scale,const Dtype* shift, int norm_size) {
  __shared__ Dtype buffer[THREAD_BLOCK_SIZE]; 
  buffer[threadIdx.x] = 0;
  if(!moving_mean) {
    for(int i = threadIdx.x; i < num * map_size; i += blockDim.x) {
      int location = i / map_size * map_size * channels + (i % map_size) + blockIdx.x * map_size;
      if(i < num * map_size)
        buffer[threadIdx.x] += pow(in[location],in_pow);
    }
    __syncthreads();
    for(int i = blockDim.x/2; i > 0; i >>= 1) {
      if(threadIdx.x < i) buffer[threadIdx.x] += buffer[threadIdx.x + i];
      __syncthreads();
    }
    if(threadIdx.x == 0) {
      buffer[0] = buffer[0] * stat_ratio;
      if(save_mean) mean[blockIdx.x] += (decay * buffer[0] + com_decay * history_mean[blockIdx.x]) / norm_size;
    }
  }
  else if(threadIdx.x == 0)
    buffer[0] = history_mean[blockIdx.x];

  __syncthreads();

  Dtype temp = pow(buffer[0] + stat_eps, stat_pow);
  Dtype scale_value = scale[blockIdx.x], shift_value = shift[blockIdx.x];
  if(threadIdx.x == 0) x_std[blockIdx.x] = temp; 
  for(int i = threadIdx.x; i < num * map_size; i += blockDim.x) {
    int location = i / map_size * map_size * channels + (i % map_size) + blockIdx.x * map_size;
    if(i < num * map_size) {
      x_norm[location] = in[location] / temp;
      out[location] = in[location] / temp * scale_value + shift_value;
    }
  }
}

template <typename Dtype>
void BNLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int num_ = bottom[0]->num();
  int channels_ = bottom[0]->channels();
  int height_ = bottom[0]->height();
  int width_ = bottom[0]->width();

  const Dtype* const_bottom_data = bottom[0]->gpu_data();
  const Dtype* const_top_data = top[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();

  const Dtype* scale_data = this->blobs_[0]->gpu_data();
  const Dtype* shift_data = this->blobs_[1]->gpu_data();
  bool save_mean = this->phase_ == TRAIN && this->param_propagate_down_[0];

  mean_statistic<Dtype><<<channels_, THREAD_BLOCK_SIZE>>>(num_, height_ * width_, channels_, 
                   Dtype(1. / (height_ * width_ * num_)),save_mean,
                   (this->phase_ == TEST || !this->param_propagate_down_[0]) && moving_average_, decay_, Dtype(1) - decay_,
                   const_bottom_data, this->blobs_[2]->mutable_gpu_diff(),
                   this->blobs_[2]->mutable_gpu_data(), top_data, Caffe::getIterSize() * Caffe::getThreadNum());
  CUDA_POST_KERNEL_CHECK;
  /*
  var_statistic<Dtype><<<channels_, THREAD_BLOCK_SIZE>>>(num_, height_ * width_, channels_, Dtype(2),
                   Dtype(1. / (height_ * width_ * num_)), var_eps_, Dtype(0.5),
                   save_mean, (this->phase_ == TEST || !this->param_propagate_down_[0]) && moving_average_,
                   (num_)*decay_/(num_-1), Dtype(1)-(num_)*decay_/(num_-1),
                   const_top_data, this->blobs_[3]->mutable_gpu_diff(),this->blobs_[3]->mutable_gpu_data(), 
                   top_data,x_norm_.mutable_gpu_data(),x_std_.mutable_gpu_data(),
                   scale_data,shift_data, Caffe::getIterSize() * Caffe::getThreadNum());
  */
  var_statistic<Dtype><<<channels_, THREAD_BLOCK_SIZE>>>(num_, height_ * width_, channels_, Dtype(2),
                   Dtype(1. / (height_ * width_ * num_)), var_eps_, Dtype(0.5),
                   save_mean, (this->phase_ == TEST || !this->param_propagate_down_[0]) && moving_average_,
                   decay_, Dtype(1)-decay_,
                   const_top_data, this->blobs_[3]->mutable_gpu_diff(),this->blobs_[3]->mutable_gpu_data(),
                   top_data,x_norm_.mutable_gpu_data(),x_std_.mutable_gpu_data(),
                   scale_data,shift_data, Caffe::getIterSize() * Caffe::getThreadNum());
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void scale_shift_bottom_gradient(const int num, const int map_size, const int channels,
    const Dtype* in, const Dtype* x_norm, Dtype* scale_diff, Dtype* shift_diff, const Dtype* scale_data,
    const Dtype* x_std, Dtype* out) {
  __shared__ Dtype buffer_scale_diff[THREAD_BLOCK_SIZE]; 
  __shared__ Dtype buffer_shift_diff[THREAD_BLOCK_SIZE]; 
  buffer_scale_diff[threadIdx.x] = 0;
  buffer_shift_diff[threadIdx.x] = 0;
  for(int i = threadIdx.x; i < num * map_size; i += blockDim.x) {
    int location = i / map_size * map_size * channels + (i % map_size) + blockIdx.x * map_size;
    if(i < num * map_size){
      buffer_scale_diff[threadIdx.x] += (in[location] * x_norm[location]);
      buffer_shift_diff[threadIdx.x] += in[location];
    }
  }
  __syncthreads();
  for(int i = blockDim.x / 2; i > 0; i >>= 1) {
    if(threadIdx.x < i) buffer_scale_diff[threadIdx.x] += buffer_scale_diff[threadIdx.x + i];
    if(threadIdx.x < i) buffer_shift_diff[threadIdx.x] += buffer_shift_diff[threadIdx.x + i];
    __syncthreads();
  }
  if(threadIdx.x == 0) {
    scale_diff[blockIdx.x] = buffer_scale_diff[0];
    shift_diff[blockIdx.x] = buffer_shift_diff[0];
  }
  __syncthreads();
  Dtype s_data_v = scale_data[blockIdx.x], x_std_v = x_std[blockIdx.x];
  for(int i = threadIdx.x; i < num * map_size; i += blockDim.x) {
    int location = i / map_size * map_size * channels + (i % map_size) + blockIdx.x * map_size;
    if(i < num * map_size) {
      out[location] = s_data_v * (in[location] - (x_norm[location] * 
          buffer_scale_diff[0] + buffer_shift_diff[0]) / (num * map_size)) / x_std_v;
    }
  }
}

template <typename Dtype>
void BNLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  int num_ = bottom[0]->num();
  int channels_ = bottom[0]->channels();
  int height_ = bottom[0]->height();
  int width_ = bottom[0]->width();

  const Dtype* const_bottom_diff = bottom[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const Dtype* const_top_diff = top[0]->gpu_diff();	
  
  Dtype* scale_diff = this->blobs_[0]->mutable_gpu_diff();
  Dtype* shift_diff = this->blobs_[1]->mutable_gpu_diff();
  const Dtype* scale_data = this->blobs_[0]->gpu_data();
  
  if (this->param_propagate_down_[0] && propagate_down[0]) {
    scale_shift_bottom_gradient<Dtype><<<channels_, THREAD_BLOCK_SIZE>>>(num_, height_ * width_, channels_,
        const_top_diff, x_norm_.gpu_data(), scale_diff, shift_diff, scale_data, x_std_.gpu_data(), bottom_diff);
    CUDA_POST_KERNEL_CHECK;
  }
  caffe_copy(this->blobs_[2]->count(), this->blobs_[2]->gpu_diff(), this->blobs_[2]->mutable_gpu_data());
  caffe_copy(this->blobs_[3]->count(), this->blobs_[3]->gpu_diff(), this->blobs_[3]->mutable_gpu_data());
}

//INSTANTIATE_CLASS(BNLayer);
INSTANTIATE_LAYER_GPU_FUNCS(BNLayer);
}  // namespace caffe
