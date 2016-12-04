#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void BatchReductionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  op_ = this->layer_param_.batch_reduction_param().reduction_param().operation();
  axis_ = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.batch_reduction_param().reduction_param().axis());

  // load levels
  int n_level = this->layer_param_.batch_reduction_param().level_size();
  if (n_level == 0) {
    this->layer_param_.mutable_batch_reduction_param()->add_level(1);
    n_level = 1;
  }
  levels_.reserve(this->layer_param_.batch_reduction_param().level_size());

  for (int i = 0; i < n_level; ++i){
    levels_.push_back(this->layer_param_.batch_reduction_param().level(i));
    ticks_.push_back(levels_.back() * levels_.back());
  }

}

template <typename Dtype>
void BatchReductionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  vector<int> top_shape(bottom[0]->shape().begin(),
                        bottom[0]->shape().begin() + axis_);

  // if levels = [1], we do global reduction instead
  if ((levels_.size() != 1) || (levels_[0] != 1)){
    top_shape.push_back(levels_.size());
    int red_dim = 0;
    for (int i = 0; i < ticks_.size(); ++i) red_dim += ticks_[i];
    CHECK_EQ(red_dim, bottom[0]->shape(axis_));
  }else{
    ticks_[0] = bottom[0]->shape(axis_); // levels=[1] means we reduce along the whole axis
  }

  for (int i = axis_ + 1; i < bottom[0]->shape().size(); ++i){
    top_shape.push_back(bottom[0]->shape()[i]);
  }
  top[0]->Reshape(top_shape);

  step_ = bottom[0]->count(axis_+1);
  num_ = bottom[0]->count(0, axis_);

  //LOG_INFO<<num_<<" "<<step_;
  CHECK_EQ(step_ * num_ * levels_.size(), top[0]->count());

  // will add these later
  if (op_ == ReductionParameter_ReductionOp_SUMSQ || op_ == ReductionParameter_ReductionOp_ASUM){
    NOT_IMPLEMENTED;
  }

  ticks_blob_.Reshape(ticks_.size(), 1, 1, 1);
  Dtype* tick_data = ticks_blob_.mutable_cpu_data();
  for (int i = 0; i < levels_.size(); ++i){
    tick_data[i] = ticks_[i];
  }
}

template <typename Dtype>
void BatchReductionLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_set(top[0]->count(), Dtype(0), top_data);
  for (int n = 0; n < num_; ++n){
    //printf(" levels: %d\n", levels_.size());
    for (int l = 0; l < levels_.size(); ++l) {
      int tick = ticks_[l];
      Dtype coeff = (op_ == ReductionParameter_ReductionOp_MEAN) ? Dtype(1)/Dtype(tick) : Dtype(1);
      for (int t = 0; t < tick; ++t) {
        caffe_cpu_axpby(step_, coeff, bottom_data, Dtype(1), top_data);
        bottom_data += step_;
      }
      top_data += step_;
    }
  }

}

template <typename Dtype>
void BatchReductionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }
  // Get bottom_data, if needed.
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  for (int i = 0; i < num_; ++i) {
    for (int l = 0; l < levels_.size(); ++l) {
      int tick = ticks_[l];
      Dtype coeff = (op_ == ReductionParameter_ReductionOp_MEAN) ? Dtype(1)/Dtype(tick) : Dtype(1);
      for (int t = 0; t < tick; ++t) {
        caffe_cpu_axpby(step_, coeff, top_diff, Dtype(0), bottom_diff);
        //offset bottom_data each input step
        bottom_diff += step_;
      }
      //offset bottom_data each output step
      top_diff += step_;
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(BatchReductionLayer);
#endif

INSTANTIATE_CLASS(BatchReductionLayer);
REGISTER_LAYER_CLASS(BatchReduction);

}  // namespace caffe
