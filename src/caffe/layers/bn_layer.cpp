#include <algorithm>
#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void BNLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  x_norm_gpu_data_ = NULL;
  x_norm_cpu_data_ = NULL;
  axis_ = this->layer_param_.bn_param().axis();
  CHECK(axis_ == 1 || axis_ == 2) << "axis_ should be 1 or 2";
  channels_ = bottom[0]->LegacyShape(axis_);

  // extract param
  var_eps_ = this->layer_param_.bn_param().var_eps();
  decay_ = this->layer_param_.bn_param().decay();
  moving_average_ = this->layer_param_.bn_param().moving_average();

  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(4);
    vector<int> shape(4, 1);
    shape[axis_] = channels_;

    // fill scale with scale_filler
    this->blobs_[0].reset(new Blob<Dtype>(shape));
    shared_ptr<Filler<Dtype> > scale_filler(GetFiller<Dtype>(
        this->layer_param_.bn_param().scale_filler()));
    scale_filler->Fill(this->blobs_[0].get());

    // fill shift with shift_filler
    this->blobs_[1].reset(new Blob<Dtype>(shape));
    shared_ptr<Filler<Dtype> > shift_filler(GetFiller<Dtype>(
        this->layer_param_.bn_param().shift_filler()));
    shift_filler->Fill(this->blobs_[1].get());

    // history mean
    this->blobs_[2].reset(new Blob<Dtype>(shape));
    caffe_set(channels_, Dtype(0), this->blobs_[2]->mutable_cpu_data());

    // history variance
    this->blobs_[3].reset(new Blob<Dtype>(shape));
    caffe_set(channels_, Dtype(0), this->blobs_[3]->mutable_cpu_data());

  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
  // runing average stats does not use weight decay and learning rate
  while (this->layer_param_.param_size() < 4){
    this->layer_param_.mutable_param()->Add();
  }
  this->layer_param_.mutable_param(2)->set_lr_mult(float(0));
  this->layer_param_.mutable_param(2)->set_decay_mult(float(0));

  this->layer_param_.mutable_param(3)->set_lr_mult(float(0));
  this->layer_param_.mutable_param(3)->set_decay_mult(float(0));
}

template <typename Dtype>
void BNLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  x_norm_gpu_data_ = NULL;
  x_norm_cpu_data_ = NULL;
  count_ = bottom[0]->count();
  // reshape blob
  top[0]->ReshapeLike(*bottom[0]);
  x_norm_.ReshapeLike(*bottom[0]);
  CHECK_EQ(channels_, bottom[0]->LegacyShape(axis_));
  vector<int> shape(4, 1);
  shape[axis_] = channels_;
  x_std_.Reshape(shape);

  // batch_statistic_
  batch_statistic_.Reshape(shape);

  // spatial_statistic_
  for (int i = 0; i < axis_; ++i)
    shape[i] = bottom[0]->LegacyShape(i);
  spatial_statistic_.Reshape(shape);

  // fill batch multiplier
  shape[axis_] = 1;
  batch_sum_multiplier_.Reshape(shape);
  Dtype* batch_multiplier_data = batch_sum_multiplier_.mutable_cpu_data();
  num_ = batch_sum_multiplier_.count();
  caffe_set(num_, Dtype(1), batch_multiplier_data);

  // buffer blob
  buffer_blob_.ReshapeLike(*bottom[0]);

  // fill spatial multiplier
  shape.assign(4, 1);
  for (int i = 3; i > axis_; --i)
    shape[i] = bottom[0]->LegacyShape(i);
  spatial_sum_multiplier_.Reshape(shape);
  Dtype* spatial_multiplier_data = spatial_sum_multiplier_.mutable_cpu_data();
  hw_ = spatial_sum_multiplier_.count();
  caffe_set(hw_, Dtype(1), spatial_multiplier_data);
}

template <typename Dtype>
void BNLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* const_bottom_data = bottom[0]->cpu_data();
  const Dtype* const_top_data = top[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();

  const Dtype* scale_data = this->blobs_[0]->cpu_data();
  const Dtype* shift_data = this->blobs_[1]->cpu_data();
  const int nc = spatial_statistic_.count();
  const int nchw = buffer_blob_.count();

  /*** BEGIN: computes variance using var(X) = sqrt( (\sum_(X - E(X))^2)/(H_*W_*N_) ) **/
  // EX across spatial
  // spatial_mean_.mutable_cpu_data = 1/(H_*W_)*const_bottom_data*spatial_sum_multiplier_
  //								  = 1/(H_*W_)*const_bottom_data*[1,1,...,1]
  //								  = 1/(H_*W_)*sum(const_bottom_data)
  caffe_cpu_gemv<Dtype>(CblasNoTrans, nc, hw_, Dtype(1. / hw_), const_bottom_data,
      spatial_sum_multiplier_.cpu_data(), Dtype(0), spatial_statistic_.mutable_cpu_data());
  // EX across batch
  // batch_mean_.mutable_cpu_data = 1./N_*spatial_mean_*batch_sum_multiplier_
  // 								= 1./N_*spatial_mean_*[1,1,...,1]
  // 								= 1./N_*sum(spatial_mean_data)
  caffe_cpu_gemv<Dtype>(CblasTrans, num_, channels_, Dtype(1. / num_), spatial_statistic_.cpu_data(),
      batch_sum_multiplier_.cpu_data(), Dtype(0), batch_statistic_.mutable_cpu_data());
  // save history batch mean
  if (this->phase_ == TRAIN) {
    caffe_cpu_axpby(batch_statistic_.count(), decay_, batch_statistic_.cpu_data(), Dtype(1) - decay_,
        this->blobs_[2]->mutable_cpu_data());
  }
  if (this->phase_ == TEST && moving_average_) {
    // use moving average mean
    caffe_copy(batch_statistic_.count(), this->blobs_[2]->cpu_data(), batch_statistic_.mutable_cpu_data());
  }
  // put mean blob into buffer_blob_
  // spatial_statistic_ = batch_sum_multiplier_*batch_statistic_
  //					  = batch_sum_multiplier_*batch_mean
  //					  = batch_sum_mean
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1, Dtype(1),
      batch_sum_multiplier_.cpu_data(), batch_statistic_.cpu_data(), Dtype(0),
      spatial_statistic_.mutable_cpu_data());
  // buffer_blob_ = (-1)*spatial_sum_multiplier_*spatial_statistic_
  // 				= (-1)*spatial_sum_multiplier_*batch_sum_mean
  //				= (-1)*total_sum_mean
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, nc, hw_, 1, Dtype(-1),
      spatial_statistic_.cpu_data(), spatial_sum_multiplier_.cpu_data(), Dtype(0),
      buffer_blob_.mutable_cpu_data());
  // substract mean
  // top_data = X - E(X)
  caffe_add(nchw, const_bottom_data, buffer_blob_.cpu_data(), top_data);

  // ---------- variance normalization ---------- //
  // put the squares of X - E(X) into buffer_blob_
  // buffer_blob_ = (X - E(X))^2
  caffe_powx(nchw, const_top_data, Dtype(2), buffer_blob_.mutable_cpu_data());
  // statistic across spatial
  // spatial_statistic_ = 1/(H_*W_)*spatial_sum_multiplier_*buffer_blob_
  caffe_cpu_gemv<Dtype>(CblasNoTrans, nc, hw_, Dtype(1. / hw_), buffer_blob_.cpu_data(),
      spatial_sum_multiplier_.cpu_data(), Dtype(0), spatial_statistic_.mutable_cpu_data());
  // statistic across batch
  // batch_statistic_ = (1/N_)*batch_sum_multiplier_*spatial_statistic_
  caffe_cpu_gemv<Dtype>(CblasTrans, num_, channels_, Dtype(1. / num_), spatial_statistic_.cpu_data(),
      batch_sum_multiplier_.cpu_data(), Dtype(0), batch_statistic_.mutable_cpu_data());
  // save history batch variance
  if (this->phase_ == TRAIN) {
    caffe_cpu_axpby(batch_statistic_.count(), decay_, batch_statistic_.cpu_data(), Dtype(1) - decay_,
        this->blobs_[3]->mutable_cpu_data());
  }
  if (this->phase_ == TEST && moving_average_) {
    // use moving average variance
    caffe_copy(batch_statistic_.count(), this->blobs_[3]->cpu_data(), batch_statistic_.mutable_cpu_data());
  }
  // add eps
  // batch_statistic_ = batch_statistic_ + var_eps_
  caffe_add_scalar(batch_statistic_.count(), var_eps_, batch_statistic_.mutable_cpu_data());
  // std
  // batch_statistic_ = sqrt(batch_statistic_)
  caffe_powx(batch_statistic_.count(), batch_statistic_.cpu_data(), Dtype(0.5),
      batch_statistic_.mutable_cpu_data());
  // put std blob into buffer_blob_
  // spatial_statistic_ = batch_sum_multiplier_ * batch_statistic_
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1, Dtype(1),
      batch_sum_multiplier_.cpu_data(), batch_statistic_.cpu_data(), Dtype(0),
      spatial_statistic_.mutable_cpu_data());
  // buffer_blob_ = spatial_statistic_ * spatial_sum_multiplier_
  // 				= all elements are equal to norm
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, nc, hw_, 1, Dtype(1),
      spatial_statistic_.cpu_data(), spatial_sum_multiplier_.cpu_data(), Dtype(0),
      buffer_blob_.mutable_cpu_data());
  // variance normalization
  // top_data = const_top_data / buffer_blob_
  // 			= const_top_data / norm
  caffe_div(nchw, const_top_data, buffer_blob_.cpu_data(), top_data);
  /*** END: computes variance using var(X) = sqrt( (\sum_(X - E(X))^2)/(H_*W_*N_) ) **/

  // ---------- save x_norm and x_std ---------- //
  caffe_copy(nchw, const_top_data, x_norm_.mutable_cpu_data());
  caffe_copy(batch_statistic_.count(), batch_statistic_.cpu_data(), x_std_.mutable_cpu_data());

  /*** BEGIN: do scale*norm + shift***/
  // ---------- scale ---------- //
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1, Dtype(1),
      batch_sum_multiplier_.cpu_data(), scale_data, Dtype(0),
      spatial_statistic_.mutable_cpu_data());
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, nc, hw_, 1, Dtype(1),
      spatial_statistic_.cpu_data(), spatial_sum_multiplier_.cpu_data(), Dtype(0),
      buffer_blob_.mutable_cpu_data());
  caffe_mul(nchw, const_top_data, buffer_blob_.cpu_data(), top_data);

  // ---------- shift ---------- //
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1, Dtype(1),
      batch_sum_multiplier_.cpu_data(), shift_data, Dtype(0),
      spatial_statistic_.mutable_cpu_data());
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, nc, hw_, 1, Dtype(1),
      spatial_statistic_.cpu_data(), spatial_sum_multiplier_.cpu_data(), Dtype(0),
      buffer_blob_.mutable_cpu_data());
  caffe_add(nchw, const_top_data, buffer_blob_.cpu_data(), top_data);
  /*** END: do scale*norm + shift ***/
}

template <typename Dtype>
void BNLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* const_bottom_diff = bottom[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* const_top_diff = top[0]->cpu_diff();

  Dtype* scale_diff = this->blobs_[0]->mutable_cpu_diff();
  Dtype* shift_diff = this->blobs_[1]->mutable_cpu_diff();
  const Dtype* scale_data = this->blobs_[0]->cpu_data();
  const int nc = spatial_statistic_.count();
  const int nchw = buffer_blob_.count();
  // ---------- gradient w.r.t. scale ---------- //
  caffe_mul(nchw, x_norm_.cpu_data(), const_top_diff, buffer_blob_.mutable_cpu_data());
  // statistic across spatial
  caffe_cpu_gemv<Dtype>(CblasNoTrans, nc, hw_, Dtype(1), buffer_blob_.cpu_data(),
      spatial_sum_multiplier_.cpu_data(), Dtype(0), spatial_statistic_.mutable_cpu_data());
  // statistic across batch
  caffe_cpu_gemv<Dtype>(CblasTrans, num_, channels_, Dtype(1), spatial_statistic_.cpu_data(),
      batch_sum_multiplier_.cpu_data(), Dtype(0), scale_diff);

  // ---------- gradient w.r.t. shift ---------- //
  // statistic across spatial
  caffe_cpu_gemv<Dtype>(CblasNoTrans, nc, hw_, Dtype(1), const_top_diff,
      spatial_sum_multiplier_.cpu_data(), Dtype(0), spatial_statistic_.mutable_cpu_data());
  // statistic across batch
  caffe_cpu_gemv<Dtype>(CblasTrans, num_, channels_, Dtype(1), spatial_statistic_.cpu_data(),
      batch_sum_multiplier_.cpu_data(), Dtype(0), shift_diff);

  // ---------- gradient w.r.t. to bottom blob ---------- //
  // put scale * top_diff to buffer_blob_
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1, Dtype(1),
      batch_sum_multiplier_.cpu_data(), scale_data, Dtype(0),
      spatial_statistic_.mutable_cpu_data());
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, nc, hw_, 1, Dtype(1),
      spatial_statistic_.cpu_data(), spatial_sum_multiplier_.cpu_data(), Dtype(0),
      buffer_blob_.mutable_cpu_data());
  caffe_mul(nchw, const_top_diff, buffer_blob_.cpu_data(), buffer_blob_.mutable_cpu_data());

  // use new top diff for computation
  caffe_mul(nchw, x_norm_.cpu_data(), buffer_blob_.cpu_data(), bottom_diff);
  // statistic across spatial
  caffe_cpu_gemv<Dtype>(CblasNoTrans, nc, hw_, Dtype(1), const_bottom_diff,
      spatial_sum_multiplier_.cpu_data(), Dtype(0), spatial_statistic_.mutable_cpu_data());
  // statistic across batch
  caffe_cpu_gemv<Dtype>(CblasTrans, num_, channels_, Dtype(1), spatial_statistic_.cpu_data(),
      batch_sum_multiplier_.cpu_data(), Dtype(0), batch_statistic_.mutable_cpu_data());

  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1, Dtype(1),
      batch_sum_multiplier_.cpu_data(), batch_statistic_.cpu_data(), Dtype(0),
      spatial_statistic_.mutable_cpu_data());
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, nc, hw_, 1, Dtype(1),
      spatial_statistic_.cpu_data(), spatial_sum_multiplier_.cpu_data(), Dtype(0),
      bottom_diff);

  caffe_mul(nchw, x_norm_.cpu_data(), const_bottom_diff, bottom_diff);

  // statistic across spatial
  caffe_cpu_gemv<Dtype>(CblasNoTrans, nc, hw_, Dtype(1), buffer_blob_.cpu_data(),
      spatial_sum_multiplier_.cpu_data(), Dtype(0), spatial_statistic_.mutable_cpu_data());
  // statistic across batch
  caffe_cpu_gemv<Dtype>(CblasTrans, num_, channels_, Dtype(1), spatial_statistic_.cpu_data(),
      batch_sum_multiplier_.cpu_data(), Dtype(0), batch_statistic_.mutable_cpu_data());

  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1, Dtype(1),
      batch_sum_multiplier_.cpu_data(), batch_statistic_.cpu_data(), Dtype(0),
      spatial_statistic_.mutable_cpu_data());
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, nc, hw_, 1, Dtype(1),
      spatial_statistic_.cpu_data(), spatial_sum_multiplier_.cpu_data(), Dtype(1),
      bottom_diff);

  caffe_cpu_axpby(nchw, Dtype(1), buffer_blob_.cpu_data(), Dtype(-1. / (num_ * hw_)),
      bottom_diff);

  // variance normalization
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1, Dtype(1),
      batch_sum_multiplier_.cpu_data(), x_std_.cpu_data(), Dtype(0),
      spatial_statistic_.mutable_cpu_data());
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, nc, hw_, 1, Dtype(1),
      spatial_statistic_.cpu_data(), spatial_sum_multiplier_.cpu_data(), Dtype(0),
      buffer_blob_.mutable_cpu_data());

  caffe_div(nchw, const_bottom_diff, buffer_blob_.cpu_data(), bottom_diff);

}

#ifdef CPU_ONLY
STUB_GPU(BNLayer);
#endif

INSTANTIATE_CLASS(BNLayer);
}  // namespace caffe
