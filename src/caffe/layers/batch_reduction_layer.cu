#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/common_layers.hpp"

namespace caffe {

// The CUDA kernel actually runs the reduction
template <typename Dtype>
__global__ void BatchReductionForwardKer(const int step, const int num,
                                         const int n_level, const Dtype* ticks,
                                         const bool mean, const bool forward,
                                         Dtype* bottom, Dtype* top) {
    Dtype* bottom_ptr = bottom;
    Dtype* top_ptr = top;
    CUDA_KERNEL_LOOP(index, step){
        for (int n = 0; n < num; ++n){
            for (int l = 0; l < n_level; ++l){
                int tick = ticks[l];
                Dtype coeff = (mean)? Dtype(1)/Dtype(tick) : Dtype(1);
                for (int t = 0; t < tick; ++t){
                    if (forward){
                        top_ptr[index] += bottom_ptr[index] * coeff;
                    }else{
                        bottom_ptr[index] = top_ptr[index] * coeff;
                    }
                    bottom_ptr += step;
                }
                top_ptr += step;
            }
        }
    }
}

template <typename Dtype>
void BatchReductionLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    const Dtype* tick_data = this->ticks_blob_.gpu_data();
    const bool kMean = (this->op_ == ReductionParameter_ReductionOp_MEAN);
    const int n_level = this->levels_.size();

    const bool kForward = true; // forward

    caffe_gpu_set(top[0]->count(), Dtype(0), top_data);
    //invoke kernel
    BatchReductionForwardKer<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
    <<<CAFFE_GET_BLOCKS(step_), CAFFE_CUDA_NUM_THREADS>>>(
        step_, num_, n_level, tick_data,
        kMean, kForward, (Dtype*)bottom_data, top_data);

}

template <typename Dtype>
void BatchReductionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    const Dtype *top_diff = top[0]->gpu_diff();
    Dtype *bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype *tick_data = this->ticks_blob_.gpu_data();
    const bool kMean = (this->op_ == ReductionParameter_ReductionOp_MEAN);
    const int n_level = this->levels_.size();

    const bool kForward = false; // backward

    //invoke kernel
    BatchReductionForwardKer<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
    <<<CAFFE_GET_BLOCKS(step_), CAFFE_CUDA_NUM_THREADS>>>(
        step_, num_, n_level, tick_data,
        kMean, kForward, bottom_diff, (Dtype*)top_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(BatchReductionLayer);

}  // namespace caffe
