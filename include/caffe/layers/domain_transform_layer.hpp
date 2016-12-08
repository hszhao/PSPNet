#ifndef CAFFE_DOMAIN_TRANSFORM_LAYER_HPP_
#define CAFFE_DOMAIN_TRANSFORM_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief The DomainTransform layer performs fast edge preserving
 * filtering, which filters the input with respect to a given
 * reference gradient.
 *
 * Domain transform preserves the geodesic distance between
 * points on the curves on 2D image manifold, adaptively
 * warping the input signal so that 1D edge-preserving
 * filtering can be efficiently performed in linear time.
 *
 * See this paper for more details:
 * Domain Transform for Edge-Aware Image and Video Processing
 *   Eduardo S. L. Gastal, and Manuel M. Oliveira
 *   SIGGRAPH 2011
 *
 * Note that domain transform performs only with respect to actual input
 * dimension (inferred from bottom[2]).
 *
 * INPUTS:
 * 0: (num, channel, height, width): input to be filtered
 * 1: (num, 1, height, width): reference gradient to be filtered against
 * 2: (num, 2, 1, 1): actual input dimension (before possible zero-padding).
 * OUTPUTS:
 * 0: (num, channel, height, width): output (filtered results)
 *
 */

template <typename Dtype>
class DomainTransformLayer : public Layer<Dtype> {
 public:
  explicit DomainTransformLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}

  virtual ~DomainTransformLayer() {
    for (int k = 0; k < intermediate_results_.size(); ++k) {
      delete intermediate_results_[k];
    }
    intermediate_results_.clear();
  }

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "DomainTransform"; }

  // Expected three inputs to be
  // (1) input to be filtered (e.g., DCNN features).
  // (2) reference gradient (to be filtered against).
  // (3) input dimenstions.
  virtual inline int ExactBottomBlobs() const { return 3; }
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

  Dtype ComputeSigma(const int iter);

  void SetUpWeightImage(const int input_height, const int input_width,
                        const Dtype* ref_grad_data, const Dtype sigma_i,
                        Dtype* weight);

  void ComputeReferenceGradientDiff(
      const int input_height, const int input_width,
       Dtype sigma_i, const Dtype* weight, const Dtype* weight_diff,
       Dtype* ref_grad_diff);

  // ForwardPass, when performing recursive Filtering from left to right.
  void HorizontalFilterLeftToRightForward(const int input_height,
       const int input_width, const Dtype* weight,
       Dtype* intermediate_res, Dtype* output);

  // BackwardPass for left to right filtering.
  void HorizontalFilterLeftToRightBackward(const int input_height,
       const int input_width, const Dtype* weight,
       const Dtype* intermediate_res, Dtype* output, Dtype* weight_diff);

  // ForwardPass, when performing recursive Filtering from right to left.
  void HorizontalFilterRightToLeftForward(const int input_height,
       const int input_width, const Dtype* weight,
       Dtype* intermediate_res, Dtype* output);

  // BackwardPass for right to left filtering.
  void HorizontalFilterRightToLeftBackward(const int input_height,
       const int input_width, const Dtype* weight,
       const Dtype* intermediate_res, Dtype* output, Dtype* weight_diff);

  // ForwardPass, when performing recursive Filtering from top to bottom.
  void VerticalFilterTopToBottomForward(const int input_height,
       const int input_width, const Dtype* weight,
       Dtype* intermediate_res, Dtype* output);

  // BackwardPass for top to bottom filtering.
  void VerticalFilterTopToBottomBackward(const int input_height,
       const int input_width, const Dtype* weight,
       const Dtype* intermediate_res, Dtype* output, Dtype* weight_diff);

  // ForwardPass, when performing recursive Filtering from bottom to top.
  void VerticalFilterBottomToTopForward(const int input_height,
       const int input_width, const Dtype* weight,
       Dtype* intermediate_res, Dtype* output);

  // BackwardPass for bottom to top filtering.
  void VerticalFilterBottomToTopBackward(const int input_height,
       const int input_width, const Dtype* weight,
       const Dtype* intermediate_res, Dtype* output, Dtype* weight_diff);

  // Spatial bandwith: standard deviation in the spatial domain.
  Dtype spatial_sigma_;
  // Range bandwith: standard deviation in the range domain.
  Dtype range_sigma_;
  // Number of outer iterations for filtering.
  int num_iter_;
  // Minimum weight value
  Dtype min_weight_;
  // Input dimensions.
  int num_;
  int channels_;
  // height_and width_ may be the padded height and width, since
  // we will pad mean pixels so that every input image has the same
  // spatial dimension (used for batch training).
  int height_;
  int width_;

  // Intermediate results during filtering.
  // The size is "num_iter_ x num_passes_", because
  // we will filter the input four times in the following orders:
  // (0) left->right (1) right->left (2) top->bottom (3) bottom->top.
  int num_passes_;
  // For backward propagation, the intermediate results during forwardpass
  // are saved.
  vector<Blob<Dtype>*> intermediate_results_;
  // weight_image_ is the weighted reference gradient (to be filtered against),
  // which depends on current iteration value, spatial_sigma_, and range_sigma_.
  Blob<Dtype> weight_image_;
  // blob_weight_diff is a temporary buffer shared for all samples. It
  // saves the gradients for weight_image, and will be used to compute the
  // gradients for reference gradient.
  Blob<Dtype> blob_weight_diff_;
};

}  // namespace caffe

#endif  // CAFFE_DOMAIN_TRANSFORM_LAYER_HPP_
