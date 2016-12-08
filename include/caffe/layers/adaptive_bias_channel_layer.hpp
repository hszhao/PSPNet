#ifndef CAFFE_ADAPTIVE_BIAS_CHANNEL_HPP_
#define CAFFE_ADAPTIVE_BIAS_CHANNEL_HPP_

#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
/**
 * @brief Adds to the input scores (specified in bottom[0]) adaptive biases in the
 * channels (listed in bottom[1]) so as they win on a target pre-defined portion
 * of the image.
 *
 * The result is only approximate (i.e., the target portions are not reached exactly).
 * In particular, the algorithm has an inner loop over the labels and at each iteration
 * ensures that the particular label wins in at least a portion_ fraction of the pixels.
 * When moving to the next labels we may lose some of the pixels that were claimed before.
 * We have an outer loop of num_iter_ iterations in which we repeat the process.
 * 
 * We always visit the background score first. To alleviate the effect of visit order
 * dependence, we iterate over the foreground classes which are present in random order.
 *
 * If suppress_others_ is true, then we make sure that the labels not listed in bottom[1]
 * score lower throughout the image in comparison to both the background and the existing
 * foreground labels.
 */
template <typename Dtype>
class AdaptiveBiasChannelLayer : public Layer<Dtype> {
 public:
  explicit AdaptiveBiasChannelLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "AdaptiveBiasChannel"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
 
  int num_, channels_, height_, width_;
  int max_labels_;

  // configurable params
  int num_iter_;
  Dtype bg_portion_, fg_portion_;
  bool suppress_others_;
  Dtype margin_others_;
};

}  // namespace caffe

#endif  // CAFFE_ADAPTIVE_BIAS_CHANNEL_HPP_
