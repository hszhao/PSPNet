#ifndef CAFFE_DENSE_CRF_LAYER_HPP_
#define CAFFE_DENSE_CRF_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/densecrf_pairwise.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief The DenseCRF layer performs mean-field inference under a
 *  fully-connected CRF model with Gaussian potentials.
 *
 */
template <typename Dtype>
class DenseCRFLayer : public Layer<Dtype> {
 public:
  explicit DenseCRFLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual ~DenseCRFLayer();

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "DenseCRF"; }
  // will take DCNN output, image (optional) and image_dim as input
  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  virtual void SetupPairwiseFunctions(const vector<Blob<Dtype>*>& bottom);
  virtual void ClearPairwiseFunctions();

  virtual void SetupUnaryEnergy(const Dtype* bottom);

  virtual void ComputeMap(Dtype* top_inf);

  virtual void RunInference();
  virtual void StartInference();
  virtual void StepInference();

  virtual void ExpAndNormalize(float* out, const float* in, float scale);

  virtual void AllocateAllData();
  virtual void DeAllocateAllData();

  
  bool has_image;

  int num_;
  int pad_height_;   // may have padded rows
  int pad_width_;    // may have padded cols

  int M_;   // number of input feature (channel)
  int W_;   // effective width   (<= pad_width_)
  int H_;   // effective height  (<= pad_height_)
  int N_;   // = W_ * H_

  int max_iter_;

  // Gaussian pairwise potential with weight and positional standard deviation
  std::vector<float> pos_w_;
  std::vector<float> pos_xy_std_;
  
  // Bilateral pairwise potential with weight, positional std, and color std
  std::vector<float> bi_w_;
  std::vector<float> bi_xy_std_;
  std::vector<float> bi_rgb_std_;

  std::vector<PairwisePotential*> pairwise_;

  int unary_element_;  // size of unary energy
  int map_element_;    // size of map result

  float* unary_;     // unary energy
  float* current_;   // current inference values, will copy to top[0]
  float* next_;      // next inference values
  float* tmp_;       // buffer

  /// sum_multiplier is used to carry out sum using BLAS
  Blob<Dtype> sum_multiplier_;
  /// scale is an intermediate Blob to hold temporary results.
  Blob<Dtype> scale_;
  /// norm_data is an intermediate Blob to hold temporary results.
  Blob<Dtype> norm_data_;

  // the output format is probability or score (score = log(probability))
  bool output_prob_;
};

}  // namespace caffe

#endif  // CAFFE_DENSE_CRF_LAYER_HPP_
