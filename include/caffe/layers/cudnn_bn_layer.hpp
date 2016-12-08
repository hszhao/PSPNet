#ifdef USE_CUDNN
#ifndef CAFFE_CUDNN_BN_LAYER_HPP_
#define CAFFE_CUDNN_BN_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/bn_layer.hpp"

#if CUDNN_VERSION_MIN(5, 0, 0)

namespace caffe {

/**
 * @brief cuDNN implementation of BNLayer.
 *        Fallback to BNLayer for CPU mode.
 */
template <typename Dtype>
class CuDNNBNLayer : public BNLayer<Dtype> {
 public:
  explicit CuDNNBNLayer(const LayerParameter& param)
      : BNLayer<Dtype>(param), handles_setup_(false) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual ~CuDNNBNLayer();

  virtual inline const char* type() const { return "BN"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  bool handles_setup_;
  cudnnHandle_t handle_;
  cudnnTensorDescriptor_t bottom_desc_;
  cudnnTensorDescriptor_t top_desc_;
  cudnnTensorDescriptor_t bn_param_desc_;

  Blob<Dtype> save_mean_;
  Blob<Dtype> save_inv_variance_;
};

}  // namespace caffe

#endif  // CAFFE_CUDNN_BN_LAYER_HPP_
#endif
#endif
