#include <algorithm>
#include <vector>

#include "caffe/layers/domain_transform_forward_only_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

using std::max;
using std::min;

template <typename Dtype>
void DomainTransformForwardOnlyLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top) {
  DomainTransformParameter param = this->layer_param_.domain_transform_param();
  spatial_sigma_  = param.spatial_sigma();
  range_sigma_    = param.range_sigma();
  num_iter_       = param.num_iter();
  min_weight_     = param.min_weight();
  CHECK_GT(spatial_sigma_, 0) << "Spatial sigma needs to be positive.";
  CHECK_GT(range_sigma_, 0)   << "Range sigma needs to be positive.";
  CHECK_GE(min_weight_, 0)    << "Minimum weight value needs to be non-negative.";
  CHECK_GT(num_iter_, 0) << "number of iteration should be larger than 0.";
  // Four passes:
  // (0) left->right (1) right->left (2) top->bottom (3) bottom->top.
  num_passes_ =  4;
}

template <typename Dtype>
void DomainTransformForwardOnlyLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top) {
  // Assume
  // bottom[0]: input to be filtered (same resolution as bottom[1]).
  // bottom[1]: reference gradient.
  // bottom[2]: input size.

  num_      = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_   = bottom[0]->height();
  width_    = bottom[0]->width();

  CHECK_EQ(bottom[1]->num(), num_) <<
      "bottom[0] and bottom[1] should have the same num.";
  CHECK_EQ(bottom[1]->channels(), 1) <<
      "Reference gradient should have only one channel.";
  CHECK_EQ(bottom[1]->height(), height_) <<
      "bottom[0] and bottom[1] should have the same height.";
  CHECK_EQ(bottom[1]->width(), width_) <<
      "bottom[0] and bottom[1] should have the same width.";

  top[0]->Reshape(num_, channels_, height_, width_);

  weight_image_.Reshape(1, 1, height_, width_);
}

template <typename Dtype>
void DomainTransformForwardOnlyLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int spatial_dim = height_ * width_;
  const int sample_dim  = channels_ * spatial_dim;

  Dtype* weight = weight_image_.mutable_cpu_data_at();

  for (int n = 0; n < num_; ++n) {
    const Dtype* feat_data = bottom[0]->cpu_data_at(n);
    Dtype* top_data        = top[0]->mutable_cpu_data_at(n);
    caffe_copy<Dtype>(sample_dim, feat_data, top_data);

    const Dtype* ref_grad_data = bottom[1]->cpu_data_at(n);
    const int input_height = static_cast<int>(bottom[2]->cpu_data_at(n)[0]);
    const int input_width  = static_cast<int>(bottom[2]->cpu_data_at(n)[1]);

    CHECK_LE(input_height, height_) <<
        "input_height should be less than or equal to height.";
    CHECK_LE(input_width, width_) <<
        "input_width should be less than or equal to width.";

    for (int iter = 0; iter < num_iter_; ++iter) {
      Dtype sigma_i = ComputeSigma(iter);

      SetUpWeightImage(input_height, input_width, ref_grad_data,
                       sigma_i, weight);

      // Perform recursive filtering for each input channel.
      for (int c = 0; c < channels_; ++c) {
        Dtype* cur_top_data = top[0]->mutable_cpu_data_at(n, c);

        // Filter the input four times in the following (forward) orders:
        // (0) left->right (1) right->left (2) top->bottom (3) bottom->top.
        for (int pass = 0; pass < num_passes_; ++pass) {
          switch (pass) {
            case 0:
              HorizontalFilterLeftToRightForward(input_height, input_width,
                                   weight, cur_top_data);
              break;
            case 1:
              HorizontalFilterRightToLeftForward(input_height, input_width,
                                   weight, cur_top_data);
              break;
            case 2:
              VerticalFilterTopToBottomForward(input_height, input_width,
                                 weight, cur_top_data);
              break;
            case 3:
              VerticalFilterBottomToTopForward(input_height, input_width,
                                 weight, cur_top_data);
              break;
          }
        }
      }
    }
  }
}


template <typename Dtype>
void DomainTransformForwardOnlyLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  /*
  if (propagate_down[2]) {
    LOG(FATAL) << this->type()
               << " Layer cannot back-propagate to image dimension.";
  }

  if (propagate_down[0] || propagate_down[1]) {
    const int spatial_dim = height_ * width_;
    const int sample_dim  = channels_ * spatial_dim;
    // weight_diff is a temporary buffer shared for all samples.
    Dtype* weight_diff = blob_weight_diff_.mutable_cpu_diff();
    Dtype* weight = weight_image_.mutable_cpu_data();

    for (int n = 0; n < num_; ++n) {
      const Dtype* top_diff       = top[0]->cpu_diff(n);
      Dtype* bottom_input_diff    = bottom[0]->mutable_cpu_diff_at(n);
      Dtype* bottom_ref_grad_diff = bottom[1]->mutable_cpu_diff_at(n);

      caffe_copy<Dtype>(sample_dim, top_diff, bottom_input_diff);
      caffe_set<Dtype>(spatial_dim, Dtype(0), bottom_ref_grad_diff);

      const Dtype* ref_grad_data = bottom[1]->cpu_data_at(n);
      const int input_height = static_cast<int>(bottom[2]->cpu_data_at(n)[0]);
      const int input_width  = static_cast<int>(bottom[2]->cpu_data_at(n)[1]);

      CHECK_LE(input_height, height_) <<
          "input_height should be less than or equal to height.";
      CHECK_LE(input_width, width_) <<
          "input_width should be less than or equal to width.";

      for (int iter = num_iter_ - 1; iter >= 0; --iter) {
        Dtype sigma_i = ComputeSigma(iter);

        SetUpWeightImage(input_height, input_width, ref_grad_data,
                         sigma_i, weight);

        caffe_set<Dtype>(spatial_dim, Dtype(0), weight_diff);

        // Perform backward recursive filtering for each input channel.
        for (int c = 0; c < channels_; ++c) {
          Dtype* input_diff = bottom[0]->mutable_cpu_diff_at(n, c);

          // Filter the input four times in the following (backward) orders:
          // (3) bottom->top (2) top->bottom (1) right->left (0) left->right.
          for (int pass = num_passes_ - 1; pass >= 0; --pass) {
            int ind = iter * num_passes_ + pass;
            Dtype* intermediate_res =
                intermediate_results_[ind]->mutable_cpu_data_at(n, c);

            switch (pass) {
              case 0:
                HorizontalFilterLeftToRightBackward(input_height, input_width,
                           weight, intermediate_res, input_diff, weight_diff);
                break;
              case 1:
                HorizontalFilterRightToLeftBackward(input_height, input_width,
                           weight, intermediate_res, input_diff, weight_diff);
                break;
              case 2:
                VerticalFilterTopToBottomBackward(input_height, input_width,
                         weight, intermediate_res, input_diff, weight_diff);
                break;
              case 3:
                VerticalFilterBottomToTopBackward(input_height, input_width,
                         weight, intermediate_res, input_diff, weight_diff);
                break;
            }
          }
        }
        ComputeReferenceGradientDiff(input_height, input_width, sigma_i,
                             weight, weight_diff, bottom_ref_grad_diff);
      }
    }
  }
  */
}

template <typename Dtype>
Dtype DomainTransformForwardOnlyLayer<Dtype>::ComputeSigma(const int iter) {
  return spatial_sigma_ * sqrt(3.) *
            pow(2., num_iter_ - (iter + 1.)) / sqrt(pow(4., num_iter_) - 1.);
}

template <typename Dtype>
void DomainTransformForwardOnlyLayer<Dtype>::SetUpWeightImage(
    const int input_height, const int input_width, const Dtype* ref_grad_data,
    const Dtype sigma_i, Dtype* weight) {
  // Division by zero has been checked in LayerSetUp.
  Dtype mult1 = -sqrt(2.) / sigma_i;
  Dtype mult2 = spatial_sigma_ / range_sigma_;
  for (int h = 0; h < input_height; ++h) {
    int ind = h * width_;
    for (int w = 0; w < input_width; ++w) {
      int pos     = ind + w;
      // weight must be [min_weight_, 1]
      //weight[pos] = min(max(static_cast<Dtype>(exp(mult1 * (1 + ref_grad_data[pos] * mult2))), 
      //		    static_cast<Dtype>(min_weight_)), Dtype(1));
      weight[pos] = static_cast<Dtype>(exp(mult1 * (1 + ref_grad_data[pos] * mult2)));
    }
  }
}

  /*
template <typename Dtype>
void DomainTransformForwardOnlyLayer<Dtype>::ComputeReferenceGradientDiff(
    const int input_height, const int input_width, Dtype sigma_i,
    const Dtype* weight, const Dtype* weight_diff, Dtype* ref_grad_diff) {
  // Division by zero has been checked in LayerSetUp.
  Dtype mult1 = -sqrt(2.) / sigma_i;
  Dtype mult2 = spatial_sigma_ / range_sigma_;
  for (int h = 0; h < input_height; ++h) {
    int ind = h * width_;
    for (int w = 0; w < input_width; ++w) {
      int pos = ind + w;
      ref_grad_diff[pos] += (mult1 * mult2 * weight_diff[pos] * weight[pos]);
    }
  }
}
  */

template <typename Dtype>
void DomainTransformForwardOnlyLayer<Dtype>::HorizontalFilterLeftToRightForward(
    const int input_height, const int input_width, const Dtype* weight, Dtype* output) {
  for (int h = 0; h < input_height; ++h) {
    int ind = h * width_;
    for (int w = 1; w < input_width; ++w) {
      int pos = ind + w;      
      //intermediate_res[pos] = output[pos - 1] - output[pos];
      output[pos] += weight[pos] * (output[pos - 1] - output[pos]);
    }
  }
}

  /*
template <typename Dtype>
void DomainTransformForwardOnlyLayer<Dtype>::HorizontalFilterLeftToRightBackward(
    const int input_height, const int input_width, const Dtype* weight,
    const Dtype* intermediate_res, Dtype* output, Dtype* weight_diff) {
  for (int h = 0; h < input_height; ++h) {
    int ind = h * width_;
    for (int w = input_width - 1; w >= 1; --w) {
      int pos    = ind + w;
      weight_diff[pos] = weight_diff[pos] + output[pos] * intermediate_res[pos];
      output[pos - 1]  = output[pos - 1] + weight[pos] * output[pos];
      output[pos]      = (1 - weight[pos]) * output[pos];
    }
  }
}
  */

template <typename Dtype>
void DomainTransformForwardOnlyLayer<Dtype>::HorizontalFilterRightToLeftForward(
    const int input_height, const int input_width, const Dtype* weight, Dtype* output) {
  for (int h = 0; h < input_height; ++h) {
    int ind = h * width_;
    for (int w = input_width - 2; w >= 0; --w) {
      int pos = ind + w;
      //intermediate_res[pos] = output[pos + 1] - output[pos];
      output[pos] += weight[pos + 1] * (output[pos + 1] - output[pos]);
    }
  }
}

  /*
template <typename Dtype>
void DomainTransformForwardOnlyLayer<Dtype>::HorizontalFilterRightToLeftBackward(
    const int input_height, const int input_width, const Dtype* weight,
    const Dtype* intermediate_res, Dtype* output, Dtype* weight_diff) {
  for (int h = 0; h < input_height; ++h) {
    int ind = h * width_;
    for (int w = 0; w < input_width - 1; ++w) {
      int pos     = ind + w;
      weight_diff[pos + 1] = weight_diff[pos + 1] +
          output[pos] * intermediate_res[pos];
      output[pos + 1]  = output[pos + 1] + weight[pos + 1] * output[pos];
      output[pos]      = (1 - weight[pos + 1]) * output[pos];
    }
  }
}
  */

template <typename Dtype>
void DomainTransformForwardOnlyLayer<Dtype>::VerticalFilterTopToBottomForward(
    const int input_height, const int input_width, const Dtype* weight, Dtype* output) {
  for (int w = 0; w < input_width; ++w) {
    for (int h = 1; h < input_height; ++h) {
      int prv_pos  = (h - 1) * width_ + w;
      int pos      = prv_pos + width_;
      //intermediate_res[pos] = output[prv_pos] - output[pos];
      output[pos] += weight[pos] * (output[prv_pos] - output[pos]);
    }
  }
}

  /*
template <typename Dtype>
void DomainTransformForwardOnlyLayer<Dtype>::VerticalFilterTopToBottomBackward(
    const int input_height, const int input_width, const Dtype* weight,
    const Dtype* intermediate_res, Dtype* output, Dtype* weight_diff) {
  for (int w = 0; w < input_width; ++w) {
    for (int h = input_height - 1; h >= 1; --h) {
      int prv_pos = (h - 1) * width_ + w;
      int pos     = prv_pos + width_;
      weight_diff[pos] = weight_diff[pos] + output[pos] * intermediate_res[pos];
      output[prv_pos]  = output[prv_pos] + weight[pos] * output[pos];
      output[pos]      = (1 - weight[pos]) * output[pos];
    }
  }
}
  */

template <typename Dtype>
void DomainTransformForwardOnlyLayer<Dtype>::VerticalFilterBottomToTopForward(
    const int input_height, const int input_width, const Dtype* weight, Dtype* output) {
  for (int w = 0; w < input_width; ++w) {
    for (int h = input_height - 2; h >= 0; --h) {
      int pos     = h * width_ + w;
      int nxt_pos = pos + width_;
      //intermediate_res[pos] = output[nxt_pos] - output[pos];
      output[pos] +=  weight[nxt_pos] * (output[nxt_pos] - output[pos]);
    }
  }
}
  /*
template <typename Dtype>
void DomainTransformForwardOnlyLayer<Dtype>::VerticalFilterBottomToTopBackward(
    const int input_height, const int input_width, const Dtype* weight,
    const Dtype* intermediate_res, Dtype* output, Dtype* weight_diff) {
  for (int w = 0; w < input_width; ++w) {
    for (int h = 0; h < input_height - 1; ++h) {
      int pos     = h * width_ + w;
      int nxt_pos = pos + width_;
      weight_diff[nxt_pos] = weight_diff[nxt_pos] +
          output[pos] * intermediate_res[pos];
      output[nxt_pos]  = output[nxt_pos] + weight[nxt_pos] * output[pos];
      output[pos]      = (1 - weight[nxt_pos]) * output[pos];
    }
  }
}
  */

#ifdef CPU_ONLY
STUB_GPU(DomainTransformForwardOnlyLayer);
#endif

INSTANTIATE_CLASS(DomainTransformForwardOnlyLayer);
REGISTER_LAYER_CLASS(DomainTransformForwardOnly);

}  // namespace caffe
