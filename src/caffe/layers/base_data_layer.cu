#include <vector>

#include "caffe/data_layers.hpp"


namespace caffe {

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // First, join the thread
  JoinPrefetchThread();
  // Reshape to loaded data.
  top[0]->ReshapeLike(this->prefetch_data_);
  // Copy the data
  caffe_copy(prefetch_data_.count(), prefetch_data_.cpu_data(),
      top[0]->mutable_gpu_data());
  if (this->output_labels_) {
    // Reshape to loaded labels.
    top[1]->ReshapeLike(prefetch_label_);
    // Copy the labels.
    caffe_copy(prefetch_label_.count(), prefetch_label_.cpu_data(),
        top[1]->mutable_gpu_data());
  }
  
  // Start a new prefetch thread
  CreatePrefetchThread();
}

/*
 * Jay add
 *
 */
template <typename Dtype>
void ImageDimPrefetchingDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // First, join the thread
  BasePrefetchingDataLayer<Dtype>::JoinPrefetchThread();
  // Copy the data
  caffe_copy(this->prefetch_data_.count(), this->prefetch_data_.cpu_data(),
       top[0]->mutable_gpu_data());
  if (this->output_labels_) {
    caffe_copy(this->prefetch_label_.count(), this->prefetch_label_.cpu_data(),
         top[1]->mutable_gpu_data());
  }
  if (output_data_dim_) {
    caffe_copy(prefetch_data_dim_.count(), prefetch_data_dim_.cpu_data(),
         top[2]->mutable_gpu_data());
  }

  // Start a new prefetch thread
  BasePrefetchingDataLayer<Dtype>::CreatePrefetchThread();
}


INSTANTIATE_LAYER_GPU_FORWARD(BasePrefetchingDataLayer);
INSTANTIATE_LAYER_GPU_FORWARD(ImageDimPrefetchingDataLayer);

}  // namespace caffe
