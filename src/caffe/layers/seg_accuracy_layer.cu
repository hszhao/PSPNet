#include <algorithm>
#include <vector>

#include "caffe/util/confusion_matrix.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/loss_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void confusion_kernel(
  const int count, const int num, const int channels, const int height, const int width,
  const Dtype* data, const Dtype* label, Dtype* confusion_matrix_buffer){
  for (int index = threadIdx.x + blockDim.x*blockIdx.x; index < count; index += blockDim.x*gridDim.x){
    int this_num = index / (height * width);
    int h = (index % (height * width)) / width;
    int w = (index % (height * width)) % width;

    int label_index = (this_num * height + h) * width + w;
    int gt_label = int(label[label_index]);
    if(gt_label >= 0 && gt_label < channels){
      int predict_label;
      if (channels > 1){
        Dtype MaxNum = data[(this_num * channels * height + h) * width + w];
        int MaxIdx = 0;
        for (int c = 1; c < channels; ++c) {
          int data_index = ((this_num * channels + c) * height + h) * width + w;
          if(data[data_index] > MaxNum){
            MaxNum = data[data_index];
            MaxIdx = c;
          }
        }
        predict_label = MaxIdx;
      }
      else{
        int data_index = (this_num * height + h) * width + w;
        predict_label = data[data_index] > 0.5 ? 1:0;
      }
      confusion_matrix_buffer[((gt_label+predict_label*channels)*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x] += 1;
    }
  }
}

template <typename Dtype>
void SegAccuracyLayer<Dtype>::confusion_kernel_cpu(
  const int count, const int num, const int channels, const int height, const int width,
  const Dtype* data, const Dtype* label, Dtype* confusion_matrix){
  
  Dtype *confusion_matrix_buffer;
  int blocks_num = num;
  int thread_num = CAFFE_CUDA_NUM_THREADS < height ? CAFFE_CUDA_NUM_THREADS : height;
  const long int mem_num = thread_num * blocks_num * channels * channels;
  cudaMalloc((void**)&confusion_matrix_buffer, sizeof(Dtype) * mem_num);
  CUDA_CHECK(cudaMemset(confusion_matrix_buffer, 0, sizeof(Dtype) * mem_num));

  confusion_kernel<Dtype><<<blocks_num, thread_num>>>(
    count, num, channels, height, width, data, label, confusion_matrix_buffer);
  CUDA_POST_KERNEL_CHECK;

  for(int j = 0; j <  channels * channels; j++){
    caffe_gpu_asum(blocks_num * thread_num, confusion_matrix_buffer + j * blocks_num * thread_num, confusion_matrix  + j);
  }
  cudaFree(confusion_matrix_buffer);
}

template void SegAccuracyLayer<float>::confusion_kernel_cpu(
  const int, const int, const int, const int, const int, const float*, const float*, float*);
template void SegAccuracyLayer<double>::confusion_kernel_cpu(
  const int, const int, const int, const int, const int, const double*, const double*, double*);
}  // namespace caffe
