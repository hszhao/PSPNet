#include <algorithm>
#include <vector>

#include "caffe/layers/domain_transform_layer.hpp"
#include "caffe/layer.hpp"
#include "caffe/common.cuh"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void kernel_horizontal_filter_left_to_right_forward(
    const int channels, const int height, const int width,
    const int input_height, const int input_width,
    const Dtype* weight, Dtype* intermediate_res, Dtype* output) {
  // One thread per row.
  CUDA_KERNEL_LOOP(ind, channels * input_height) {
    int h = ind % input_height;
    int c = ind / input_height;
    for (int w = 1; w < input_width; ++w) {
      int ind_out = (c * height + h) * width + w;
      int ind_wei = h * width + w;
      intermediate_res[ind_out] = output[ind_out - 1] - output[ind_out];
      output[ind_out] += weight[ind_wei] * intermediate_res[ind_out];
    }
  }
}

template <typename Dtype>
__global__ void kernel_horizontal_filter_left_to_right_backward(
    const int channels, const int height, const int width,
    const int input_height, const int input_width,
    const Dtype* weight, const Dtype* intermediate_res,
    Dtype* output, Dtype* weight_diff) {
  // One thread per row.
  CUDA_KERNEL_LOOP(ind, channels * input_height) {
    int h = ind % input_height;
    int c = ind / input_height;
    for (int w = input_width - 1; w >= 1; --w) {
      int ind_out = (c * height + h) * width + w;
      int ind_wei = h * width + w;
      atomicAdd(&weight_diff[ind_wei],
          output[ind_out] * intermediate_res[ind_out]);
      output[ind_out - 1] += weight[ind_wei] * output[ind_out];
      output[ind_out] *= 1 - weight[ind_wei];
    }
  }
}

template <typename Dtype>
__global__ void kernel_horizontal_filter_right_to_left_forward(
    const int channels, const int height, const int width,
    const int input_height, const int input_width,
    const Dtype* weight, Dtype* intermediate_res, Dtype* output) {
  // One thread per row.
  CUDA_KERNEL_LOOP(ind, channels * input_height) {
    int h = ind % input_height;
    int c = ind / input_height;
    for (int w = input_width - 2; w >= 0; --w) {
      int ind_out = (c * height + h) * width + w;
      int ind_wei = h * width + w;
      intermediate_res[ind_out] = output[ind_out + 1] - output[ind_out];
      output[ind_out] += weight[ind_wei + 1] * intermediate_res[ind_out];
    }
  }
}

template <typename Dtype>
__global__ void kernel_horizontal_filter_right_to_left_backward(
    const int channels, const int height, const int width,
    const int input_height, const int input_width,
    const Dtype* weight, const Dtype* intermediate_res,
    Dtype* output, Dtype* weight_diff) {
  // One thread per row.
  CUDA_KERNEL_LOOP(ind, channels * input_height) {
    int h = ind % input_height;
    int c = ind / input_height;
    for (int w = 0; w < input_width - 1; ++w) {
      int ind_out = (c * height + h) * width + w;
      int ind_wei = h * width + w;
      atomicAdd(&weight_diff[ind_wei + 1],
                output[ind_out] * intermediate_res[ind_out]);
      output[ind_out + 1]  += weight[ind_wei + 1] * output[ind_out];
      output[ind_out] *= 1 - weight[ind_wei + 1];
    }
  }
}

template <typename Dtype>
__global__ void kernel_vertical_filter_top_to_bottom_forward(
    const int channels, const int height, const int width,
    const int input_height, const int input_width,
    const Dtype* weight, Dtype* intermediate_res, Dtype* output) {
  // One thread per column.
  CUDA_KERNEL_LOOP(ind, channels * input_width) {
    int w = ind % input_width;
    int c = ind / input_width;
    for (int h = 1; h < input_height; ++h) {
      int ind_out = (c * height + h) * width + w;
      int ind_wei = h * width + w;
      intermediate_res[ind_out] = output[ind_out - width] - output[ind_out];
      output[ind_out] += weight[ind_wei] * intermediate_res[ind_out];
    }
  }
}

template <typename Dtype>
__global__ void kernel_vertical_filter_top_to_bottom_backward(
    const int channels, const int height, const int width,
    const int input_height, const int input_width,
    const Dtype* weight, const Dtype* intermediate_res,
    Dtype* output, Dtype* weight_diff) {
  // One thread per column.
  CUDA_KERNEL_LOOP(ind, channels * input_width) {
    int w = ind % input_width;
    int c = ind / input_width;
    for (int h = input_height - 1; h >= 1; --h) {
      int ind_out = (c * height + h) * width + w;
      int ind_wei = h * width + w;
      atomicAdd(&weight_diff[ind_wei],
                output[ind_out] * intermediate_res[ind_out]);
      output[ind_out - width]  += weight[ind_wei] * output[ind_out];
      output[ind_out] = (1 - weight[ind_wei]) * output[ind_out];
    }
  }
}

template <typename Dtype>
__global__ void kernel_vertical_filter_bottom_to_top_forward(
    const int channels, const int height, const int width,
    const int input_height, const int input_width,
    const Dtype* weight, Dtype* intermediate_res, Dtype* output) {
  // One thread per column.
  CUDA_KERNEL_LOOP(ind, channels * input_width) {
    int w = ind % input_width;
    int c = ind / input_width;
    for (int h = input_height - 2; h >= 0; --h) {
      int ind_out = (c * height + h) * width + w;
      int ind_wei = h * width + w;
      intermediate_res[ind_out] = output[ind_out + width] - output[ind_out];
      output[ind_out] += weight[ind_wei + width] * intermediate_res[ind_out];
    }
  }
}

template <typename Dtype>
__global__ void kernel_vertical_filter_bottom_to_top_backward(
    const int channels, const int height, const int width,
    const int input_height, const int input_width,
    const Dtype* weight, const Dtype* intermediate_res,
    Dtype* output, Dtype* weight_diff) {
  // One thread per column.
  CUDA_KERNEL_LOOP(ind, channels * input_width) {
    int w = ind % input_width;
    int c = ind / input_width;
    for (int h = 0; h < input_height - 1; ++h) {
      int ind_out = (c * height + h) * width + w;
      int ind_wei = h * width + w;
      atomicAdd(&weight_diff[ind_wei + width],
                output[ind_out] * intermediate_res[ind_out]);
      output[ind_out + width] += weight[ind_wei + width] * output[ind_out];
      output[ind_out] *= 1 - weight[ind_wei + width];
    }
  }
}


template <typename Dtype>
__global__ void kernel_setup_weight_image(
    const int count, const int input_width, const int width,
    const Dtype sigma_i, const Dtype spatial_sigma, const Dtype range_sigma,
    const Dtype min_weight, const Dtype* data, Dtype* weight) {
  // Division by zero has been checked in LayerSetUp.
  Dtype mult1 = -sqrt(2.) / sigma_i;
  Dtype mult2 = spatial_sigma / range_sigma;
  CUDA_KERNEL_LOOP(index, count) {
    int h   = index / input_width;
    int w   = index % input_width;
    int pos = h * width + w;
    // weight must be [min_weight_, 1]
    weight[pos] = min(max(exp(mult1 * (1 + data[pos] * mult2)), min_weight), Dtype(1));
  }
}

template <typename Dtype>
__global__ void kernel_compute_ref_grad_diff(
    const int count, const int input_width, const int width,
    const Dtype sigma_i, const Dtype spatial_sigma, const Dtype range_sigma,
    const Dtype* weight, const Dtype* weight_diff, Dtype* ref_grad_diff) {
  // Division by zero has been checked in LayerSetUp.
  Dtype mult1 = -sqrt(2.) / sigma_i;
  Dtype mult2 = spatial_sigma / range_sigma;
  CUDA_KERNEL_LOOP(index, count) {
    int h   = index / input_width;
    int w   = index % input_width;
    int pos = h * width + w;
    ref_grad_diff[pos] += (mult1 * mult2 * weight_diff[pos] * weight[pos]);
  }
}


template <typename Dtype>
void DomainTransformLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int spatial_dim = height_ * width_;
  const int sample_dim  = channels_ * spatial_dim;

  Dtype* weight = weight_image_.mutable_gpu_data();

  for (int n = 0; n < num_; ++n) {
    const Dtype* feat_data     = bottom[0]->gpu_data_at(n);
    Dtype* top_data            = top[0]->mutable_gpu_data_at(n);
    caffe_copy<Dtype>(sample_dim, feat_data, top_data);

    const Dtype* ref_grad_data = bottom[1]->gpu_data_at(n);
    const int input_height = static_cast<int>(bottom[2]->cpu_data_at(n)[0]);
    const int input_width  = static_cast<int>(bottom[2]->cpu_data_at(n)[1]);
    const int input_spatial_dim = input_height * input_width;

    CHECK_LE(input_height, height_) <<
        "input_height should be less than or equal to height.";
    CHECK_LE(input_width, width_) <<
        "input_width should be less than or equal to width.";

    for (int iter = 0; iter < num_iter_; ++iter) {
      Dtype sigma_i = ComputeSigma(iter);

      kernel_setup_weight_image<Dtype><<<CAFFE_GET_BLOCKS(
          input_spatial_dim), CAFFE_CUDA_NUM_THREADS>>>(
              input_spatial_dim, input_width, width_,
              sigma_i, spatial_sigma_, range_sigma_, min_weight_,
              ref_grad_data, weight);
      /* TODO(gpapan): This CUDA implementation is inefficient, because there
       * are dependencies within each row or col, so you can only use height
       * or width threads. You can improve this by doing all channels in
       * parallel and also being more careful with your <<< . >>> arguments.
       * You can further significantly improve speed by using BLAS *axpby()
       * routines. Right now caffe_gpu_axpby is not sufficient because it
       * assumes strides = 1, but you need to use the full BLAS interface
       * that allows strides > 1.
       * Overload caffe_gpu_axpby(), also supplying a version that accepts
       * a stride parameter. Use this to significantly improve speed. Also
       * adding this functionality to caffe_cpu_axpby() would further allow
       * you to have almost identical cpu / gpu implementations.
       */

      // Filter the input four times in the following (forward) orders:
      // (0) left->right (1) right->left (2) top->bottom (3) bottom->top.
      for (int pass = 0; pass < num_passes_; ++pass) {
        int ind = iter * num_passes_ + pass;
        Dtype* intermediate_res =
          intermediate_results_[ind]->mutable_gpu_data_at(n);

        switch (pass) {
        case 0:
          kernel_horizontal_filter_left_to_right_forward<Dtype><<<
              CAFFE_GET_BLOCKS(channels_ * input_height),
              CAFFE_CUDA_NUM_THREADS>>>(
                  channels_, height_, width_,
                  input_height, input_width,
                  weight, intermediate_res, top_data);
          break;
        case 1:
          kernel_horizontal_filter_right_to_left_forward<Dtype><<<
              CAFFE_GET_BLOCKS(channels_ * input_height),
              CAFFE_CUDA_NUM_THREADS>>>(
                  channels_, height_, width_,
                  input_height, input_width,
                  weight, intermediate_res, top_data);
          break;
        case 2:
          kernel_vertical_filter_top_to_bottom_forward<Dtype><<<
              CAFFE_GET_BLOCKS(channels_ * input_width),
              CAFFE_CUDA_NUM_THREADS>>>(
                  channels_, height_, width_,
                  input_height, input_width,
                  weight, intermediate_res, top_data);
          break;
        case 3:
          kernel_vertical_filter_bottom_to_top_forward<Dtype><<<
            CAFFE_GET_BLOCKS(channels_ * input_width),
              CAFFE_CUDA_NUM_THREADS>>>(
                  channels_, height_, width_,
                  input_height, input_width,
                  weight, intermediate_res, top_data);
          break;
        }
      }
    }
  }
}

template <typename Dtype>
void DomainTransformLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[2]) {
    LOG(FATAL) << this->type()
               << " Layer cannot back-propagate to image dimension.";
  }

  if (propagate_down[0] || propagate_down[1]) {
    const int spatial_dim = height_ * width_;
    const int sample_dim  = channels_ * spatial_dim;
    // weight_diff is a temporary buffer shared for all samples.
    Dtype* weight_diff = blob_weight_diff_.mutable_gpu_diff();
    Dtype* weight = weight_image_.mutable_gpu_data();

    for (int n = 0; n < num_; ++n) {
      const Dtype* top_diff       = top[0]->gpu_diff_at(n);
      Dtype* bottom_input_diff    = bottom[0]->mutable_gpu_diff_at(n);
      Dtype* bottom_ref_grad_diff = bottom[1]->mutable_gpu_diff_at(n);

      caffe_copy<Dtype>(sample_dim, top_diff, bottom_input_diff);
      caffe_gpu_set<Dtype>(spatial_dim, Dtype(0), bottom_ref_grad_diff);

      const Dtype* ref_grad_data  = bottom[1]->gpu_data_at(n);
      const int input_height = static_cast<int>(bottom[2]->cpu_data_at(n)[0]);
      const int input_width  = static_cast<int>(bottom[2]->cpu_data_at(n)[1]);

      CHECK_LE(input_height, height_) <<
          "input_height should be less than or equal to height.";
      CHECK_LE(input_width, width_) <<
          "input_width should be less than or equal to width.";

      const int input_spatial_dim = input_height * input_width;

      for (int iter = num_iter_ - 1; iter >= 0; --iter) {
        Dtype sigma_i = ComputeSigma(iter);

        kernel_setup_weight_image<Dtype><<<CAFFE_GET_BLOCKS(
            input_spatial_dim), CAFFE_CUDA_NUM_THREADS>>>(
                input_spatial_dim, input_width, width_,
                sigma_i, spatial_sigma_, range_sigma_, min_weight_,
                ref_grad_data, weight);

        caffe_gpu_set<Dtype>(spatial_dim, Dtype(0), weight_diff);

        /* TODO(gpapan): This CUDA implementation is inefficient, because there
         * are dependencies within each row or col, so you can only use height
         * or width threads. You can improve this by doing all channels in
         * parallel and also being more careful with your <<< . >>> arguments.
         * You can further significantly improve speed by using BLAS *axpby()
         * routines. Right now caffe_gpu_axpby is not sufficient because it
         * assumes strides = 1, but you need to use the full BLAS interface
         * that allows strides > 1.
         * Overload caffe_gpu_axpby(), also supplying a version that accepts
         * a stride parameter. Use this to significantly improve speed. Also
         * adding this functionality to caffe_cpu_axpby() would further allow
         * you to have almost identical cpu / gpu implementations.
         */

        // Filter the input four times in the following (backward) orders:
        // (3) bottom->top (2) top->bottom (1) right->left (0) left->right.
        for (int pass = num_passes_ - 1; pass >= 0; --pass) {
          int ind = iter * num_passes_ + pass;
          Dtype* intermediate_res =
            intermediate_results_[ind]->mutable_gpu_data_at(n);

          switch (pass) {
          case 0:
            kernel_horizontal_filter_left_to_right_backward<Dtype><<<
                CAFFE_GET_BLOCKS(channels_ * input_height),
                CAFFE_CUDA_NUM_THREADS>>>(
                  channels_, height_, width_,
                  input_height, input_width,
                  weight, intermediate_res, bottom_input_diff, weight_diff);
            break;
          case 1:
            kernel_horizontal_filter_right_to_left_backward<Dtype><<<
                CAFFE_GET_BLOCKS(channels_ * input_height),
                CAFFE_CUDA_NUM_THREADS>>>(
                  channels_, height_, width_,
                  input_height, input_width,
                  weight, intermediate_res, bottom_input_diff, weight_diff);
            break;
          case 2:
            kernel_vertical_filter_top_to_bottom_backward<Dtype><<<
                CAFFE_GET_BLOCKS(channels_ * input_width),
                CAFFE_CUDA_NUM_THREADS>>>(
                  channels_, height_, width_,
                  input_height, input_width,
                  weight, intermediate_res, bottom_input_diff, weight_diff);
            break;
          case 3:
            kernel_vertical_filter_bottom_to_top_backward<Dtype><<<
                CAFFE_GET_BLOCKS(channels_ * input_width),
                CAFFE_CUDA_NUM_THREADS>>>(
                  channels_, height_, width_,
                  input_height, input_width,
                  weight, intermediate_res, bottom_input_diff, weight_diff);
            break;
          }
        }
        
        kernel_compute_ref_grad_diff<Dtype><<<
            CAFFE_GET_BLOCKS(input_spatial_dim), CAFFE_CUDA_NUM_THREADS>>>(
                input_spatial_dim, input_width, width_,
                sigma_i, spatial_sigma_, range_sigma_,
                weight, weight_diff, bottom_ref_grad_diff);
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(DomainTransformLayer);

}  // namespace caffe
