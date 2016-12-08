#ifndef CAFFE_BIAS_CHANNEL_LAYER_HPP_
#define CAFFE_BIAS_CHANNEL_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
/**
 * @brief Adds bg_bias to the scores of the background channel and
 * fg_bias to the scores of each of the foreground channels
 */
template <typename Dtype>
class BiasChannelLayer : public Layer<Dtype> {
 public:
  explicit BiasChannelLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "BiasChannel"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  
  int num_, channels_, height_, width_;
  int max_labels_;
  Dtype bg_bias_, fg_bias_;
  // set of ignore labels
  std::set<int> ignore_label_;
  bool has_background_label_;
  int background_label_;
};

}  // namespace caffe

#endif  // CAFFE_BIAS_CHANNEL_LAYER_HPP_
