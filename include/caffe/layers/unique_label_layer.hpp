#ifndef CAFFE_UNIQUE_LABEL_LAYER_HPP_
#define CAFFE_UNIQUE_LABEL_LAYER_HPP_

#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
/**
 * @brief Finds the unique labels in (num, 1, height, width) input discarding
 * the positions, resulting into a (num, max_labels, 1, 1) summary output
 */
template <typename Dtype>
class UniqueLabelLayer : public Layer<Dtype> {
 public:
  explicit UniqueLabelLayer(const LayerParameter& param)
    : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "UniqueLabel"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
     const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
 
  int num_, channels_, height_, width_;
  int max_labels_;
  // set of ignore labels
  std::set<Dtype> ignore_label_;
  // set of forced labels
  std::set<Dtype> force_label_;
};

}  // namespace caffe

#endif  // CAFFE_UNIQUE_LABEL_LAYER_HPP_
