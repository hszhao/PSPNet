#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"


// TODO: check we should clear confusion_matrix somewhere!

namespace caffe {

template <typename Dtype>
void SegAccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  confusion_matrix_.clear();
  //edit by li
  if (bottom[0]->channels() >1)
    confusion_matrix_.resize(bottom[0]->channels());
  else
    confusion_matrix_.resize(2);

  SegAccuracyParameter seg_accuracy_param = this->layer_param_.seg_accuracy_param();
  for (int c = 0; c < seg_accuracy_param.ignore_label_size(); ++c){
    ignore_label_.insert(seg_accuracy_param.ignore_label(c));
  }

  output_label_.clear();
  for (int i = 0; i < seg_accuracy_param.output_label_size(); ++i){
    int output_label_one = seg_accuracy_param.output_label(i);
    CHECK_GE(output_label_one, 0);
    if (bottom[0]->channels() >1)
      CHECK_LT(output_label_one, bottom[0]->channels());
    else
      CHECK_LT(output_label_one, 2);
    output_label_.push_back(output_label_one);
  }
  CHECK_EQ(top.size(), output_label_.size() + 1);
}

template <typename Dtype>
void SegAccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_LE(1, bottom[0]->channels())
      << "top_k must be less than or equal to the number of channels (classes).";
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "The data and label should have the same number.";
  CHECK_EQ(bottom[1]->channels(), 1)
      << "The label should have one channel.";
  CHECK_EQ(bottom[0]->height(), bottom[1]->height())
      << "The data should have the same height as label.";
  CHECK_EQ(bottom[0]->width(), bottom[1]->width())
      << "The data should have the same width as label.";
 
  for (size_t i = 0; i < top.size(); ++i){
    top[i]->Reshape(1, 1, 1, 3);
  }
}

template <typename Dtype>
void SegAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();

  int data_index, label_index;

  confusion_matrix_.clear();  

  for (int i = 0; i < num; ++i) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        if (channels > 1){
          Dtype MaxNum = bottom_data[h * width + w];
          int MaxIdx = 0;
          for (int c = 1; c < channels; ++c) {
            data_index = (c * height + h) * width + w;
            if(bottom_data[data_index] > MaxNum){
              MaxNum = bottom_data[data_index];
              MaxIdx = c;
            }
          }
          const int predict_label = MaxIdx;

          label_index = h * width + w;
          const int gt_label = static_cast<int>(bottom_label[label_index]);

          if (ignore_label_.count(gt_label) != 0) {
            continue; // ignore the pixel with this gt_label
          } else if (gt_label >= 0 && gt_label < channels) {
            confusion_matrix_.accumulate(gt_label, predict_label);
          } else {
            LOG(FATAL) << "Unexpected label " << gt_label;
          }
        } 
        else {
          label_index = h * width + w;
          data_index = h* width + w;
          const int gt_label = static_cast<int>(bottom_label[label_index]);
          if (ignore_label_.count(gt_label) != 0){ 
            continue; // ignore the pixel with this gt_label
          }
          else if (gt_label ==0 || gt_label ==1){
            confusion_matrix_.accumulate(gt_label, (bottom_data[data_index] > 0.5 ? 1:0));
          }
          else {
            LOG(FATAL) << "Unexpected label " << gt_label;
          }
        }
      }
    }
    bottom_data  += bottom[0]->offset(1);
    bottom_label += bottom[1]->offset(1);
  }

  // we report all the resuls
  top[0]->mutable_cpu_data()[0] = (Dtype)confusion_matrix_.accuracy();
  top[0]->mutable_cpu_data()[1] = (Dtype)confusion_matrix_.avgRecall(false);
  top[0]->mutable_cpu_data()[2] = (Dtype)confusion_matrix_.avgJaccard();
  for (size_t i = 0; i < output_label_.size(); ++i){
    top[i+1]->mutable_cpu_data()[0] = (Dtype)confusion_matrix_.precision(output_label_[i]);
    top[i+1]->mutable_cpu_data()[1] = (Dtype)confusion_matrix_.recall(output_label_[i]);
    top[i+1]->mutable_cpu_data()[2] = (Dtype)confusion_matrix_.jaccard(output_label_[i]);
  }
}

template <typename Dtype>
void SegAccuracyLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_label = bottom[1]->gpu_data();
  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  
  Blob<Dtype> confusion;
  confusion.Reshape(1,1,channels,channels);
  confusion_kernel_cpu(num*height*width, num, channels, height, width, bottom_data, bottom_label, confusion.mutable_cpu_data());

  confusion_matrix_.clear();
  for(int i = 0; i < channels; i++){
    if(ignore_label_.count(i) != 0)
      continue;
    for(int j = 0; j < channels; j++){
      confusion_matrix_.accumulate(i, j, (unsigned long)confusion.cpu_data()[j * channels + i]);
    }
  }

  // we report all the resuls
  top[0]->mutable_cpu_data()[0] = (Dtype)confusion_matrix_.accuracy();
  top[0]->mutable_cpu_data()[1] = (Dtype)confusion_matrix_.avgRecall(false);
  top[0]->mutable_cpu_data()[2] = (Dtype)confusion_matrix_.avgJaccard();
  for (size_t i = 0; i < output_label_.size(); ++i){
    top[i+1]->mutable_cpu_data()[0] = (Dtype)confusion_matrix_.precision(output_label_[i]);
    top[i+1]->mutable_cpu_data()[1] = (Dtype)confusion_matrix_.recall(output_label_[i]);
    top[i+1]->mutable_cpu_data()[2] = (Dtype)confusion_matrix_.jaccard(output_label_[i]);
  }
}

INSTANTIATE_CLASS(SegAccuracyLayer);
REGISTER_LAYER_CLASS(SegAccuracy);
}  // namespace caffe
