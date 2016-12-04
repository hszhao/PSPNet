#include <algorithm>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"

#include "caffe/common_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

#define BATCH_SIZE 2
#define INPUT_DATA_SIZE 3

namespace caffe {

template <typename TypeParam>
class BNLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  BNLayerTest()
      : blob_bottom_(new Blob<Dtype>(5, 2, 3, 4)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~BNLayerTest() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(BNLayerTest, TestDtypesAndDevices);

TYPED_TEST(BNLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;

  BNParameter* bn_param = layer_param.mutable_bn_param();
  FillerParameter *slope_param = bn_param->mutable_slope_filler();
  slope_param->set_value(1);
  FillerParameter *bias_param = bn_param->mutable_bias_filler();
  bias_param->set_value(0);
  bn_param->set_eps(0.);
  bn_param->set_frozen(false);

  BNLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  // Test mean
  int num = this->blob_bottom_->num();
  int channels = this->blob_bottom_->channels();
  int height = this->blob_bottom_->height();
  int width = this->blob_bottom_->width();

  for (int j = 0; j < channels; ++j) {
    Dtype sum = 0, var = 0;
    for (int i = 0; i < num; ++i) {
      for ( int k = 0; k < height; ++k ) {
        for ( int l = 0; l < width; ++l ) {
          Dtype data = this->blob_top_->data_at(i, j, k, l);
          sum += data;
          var += data * data;
        }
      }
    }
    sum /= height * width * num;
    var /= height * width * num;

    const Dtype kErrorBound = 0.001;
    // expect zero mean
    EXPECT_NEAR(0, sum, kErrorBound);
    // expect unit variance
    EXPECT_NEAR(1, var, kErrorBound);
  }
}

TYPED_TEST(BNLayerTest, TestBackward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;

  BNParameter* bn_param = layer_param.mutable_bn_param();
  FillerParameter *slope_param = bn_param->mutable_slope_filler();
  slope_param->set_value(1);
  FillerParameter *bias_param = bn_param->mutable_bias_filler();
  bias_param->set_value(0);
  bn_param->set_eps(0.);
  bn_param->set_frozen(false);

  BNLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-4);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(BNLayerTest, TestForwardFrozen) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;

  BNParameter* bn_param = layer_param.mutable_bn_param();
  FillerParameter *slope_param = bn_param->mutable_slope_filler();
  slope_param->set_value(1);
  FillerParameter *bias_param = bn_param->mutable_bias_filler();
  bias_param->set_value(0);
  bn_param->set_eps(0.);
  bn_param->set_frozen(true);

  BNLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  // Init running mean and var
  shared_ptr<Blob<Dtype> > running_mean = layer.blobs()[2];
  Dtype* running_mean_data = running_mean->mutable_cpu_data();
  for (int c = 0; c < running_mean->count(); ++c) {
    running_mean_data[c] = Dtype(c);
  }
  shared_ptr<Blob<Dtype> > running_var = layer.blobs()[3];
  Dtype* running_var_data = running_var->mutable_cpu_data();
  for (int c = 0; c < running_var->count(); ++c) {
    running_var_data[c] = Dtype(c + 1);
  }

  layer.Reshape(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  int num = this->blob_bottom_->num();
  int channels = this->blob_bottom_->channels();
  int height = this->blob_bottom_->height();
  int width = this->blob_bottom_->width();

  const Dtype kErrorBound = 0.001;
  for (int j = 0; j < channels; ++j) {
    for (int i = 0; i < num; ++i) {
      for ( int k = 0; k < height; ++k ) {
        for ( int l = 0; l < width; ++l ) {
          Dtype input = this->blob_bottom_->data_at(i, j, k, l);
          Dtype output = this->blob_top_->data_at(i, j, k, l);
          Dtype expect_output = (input - j) / sqrt(j + 1);
          EXPECT_NEAR(expect_output, output, kErrorBound);
        }
      }
    }
  }
}

TYPED_TEST(BNLayerTest, TestBackwardFrozen) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;

  BNParameter* bn_param = layer_param.mutable_bn_param();
  FillerParameter *slope_param = bn_param->mutable_slope_filler();
  slope_param->set_value(1);
  FillerParameter *bias_param = bn_param->mutable_bias_filler();
  bias_param->set_value(0);
  bn_param->set_eps(0.);
  bn_param->set_frozen(true);

  BNLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  // Init running mean and var
  shared_ptr<Blob<Dtype> > running_mean = layer.blobs()[2];
  Dtype* running_mean_data = running_mean->mutable_cpu_data();
  for (int c = 0; c < running_mean->count(); ++c) {
    running_mean_data[c] = Dtype(c);
  }
  shared_ptr<Blob<Dtype> > running_var = layer.blobs()[3];
  Dtype* running_var_data = running_var->mutable_cpu_data();
  for (int c = 0; c < running_var->count(); ++c) {
    running_var_data[c] = Dtype(c + 1);
  }

  layer.Reshape(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  Dtype* top_diff_data = this->blob_top_->mutable_cpu_diff();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    top_diff_data[i] = Dtype(i);
  }
  vector<bool> propagate_down(1, true);
  layer.Backward(this->blob_top_vec_, propagate_down, this->blob_bottom_vec_);

  int num = this->blob_bottom_->num();
  int channels = this->blob_bottom_->channels();
  int height = this->blob_bottom_->height();
  int width = this->blob_bottom_->width();

  const Dtype kErrorBound = 0.001;
  for (int j = 0; j < channels; ++j) {
    for (int i = 0; i < num; ++i) {
      for ( int k = 0; k < height; ++k ) {
        for ( int l = 0; l < width; ++l ) {
          Dtype input = this->blob_top_->diff_at(i, j, k, l);
          Dtype output = this->blob_bottom_->diff_at(i, j, k, l);
          Dtype expect_output = input / sqrt(j + 1);
          EXPECT_NEAR(expect_output, output, kErrorBound);
        }
      }
    }
  }
}

#ifdef USE_CUDNN
template <typename Dtype>
class CuDNNBNLayerTest : public GPUDeviceTest<Dtype> {
 protected:
  CuDNNBNLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 4, 5)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_mean(-10);
    filler_param.set_std(5);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~CuDNNBNLayerTest() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(CuDNNBNLayerTest, TestDtypes);

TYPED_TEST(CuDNNBNLayerTest, TestForward) {
  Caffe::set_random_seed(1701);
  typedef TypeParam Dtype;
  LayerParameter layer_param;
  BNParameter* bn_param = layer_param.mutable_bn_param();
  FillerParameter *slope_param = bn_param->mutable_slope_filler();
  slope_param->set_value(1);
  FillerParameter *bias_param = bn_param->mutable_bias_filler();
  bias_param->set_value(0);
  bn_param->set_eps(0.);
  bn_param->set_frozen(false);

  CuDNNBNLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  // Test mean
  Dtype mean, var;
  int num = this->blob_bottom_->num();
  int channels = this->blob_bottom_->channels();
  int height = this->blob_bottom_->height();
  int width = this->blob_bottom_->width();

  for (int j = 0; j < channels; ++j) {
    Dtype mean = 0, var = 0;
    for (int i = 0; i < num; ++i) {
      for (int k = 0; k < height; ++k) {
        for (int l = 0; l < width; ++l) {
          Dtype data = this->blob_top_->data_at(i, j, k, l);
          mean += data;
          var += data * data;
        }
      }
    }
    mean /= num * height * width;
    var /= num * height * width;

    const Dtype kErrorBound = 0.001;
    EXPECT_NEAR(0, mean, kErrorBound);
    EXPECT_NEAR(1, var, kErrorBound);
  }
}

TYPED_TEST(CuDNNBNLayerTest, TestGradient) {
  typedef TypeParam Dtype;
  LayerParameter layer_param;
  BNParameter* bn_param = layer_param.mutable_bn_param();
  FillerParameter *slope_param = bn_param->mutable_slope_filler();
  slope_param->set_value(1);
  FillerParameter *bias_param = bn_param->mutable_bias_filler();
  bias_param->set_value(0);
  bn_param->set_eps(0.);
  bn_param->set_frozen(false);

  CuDNNBNLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 4e-4);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(CuDNNBNLayerTest, TestForwardFrozen) {
  typedef TypeParam Dtype;
  LayerParameter layer_param;

  BNParameter* bn_param = layer_param.mutable_bn_param();
  FillerParameter *slope_param = bn_param->mutable_slope_filler();
  slope_param->set_value(1);
  FillerParameter *bias_param = bn_param->mutable_bias_filler();
  bias_param->set_value(0);
  bn_param->set_eps(0.);
  bn_param->set_frozen(true);

  CuDNNBNLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  // Init running mean and var
  shared_ptr<Blob<Dtype> > running_mean = layer.blobs()[2];
  Dtype* running_mean_data = running_mean->mutable_cpu_data();
  for (int c = 0; c < running_mean->count(); ++c) {
    running_mean_data[c] = Dtype(c);
  }
  shared_ptr<Blob<Dtype> > running_var = layer.blobs()[3];
  Dtype* running_var_data = running_var->mutable_cpu_data();
  for (int c = 0; c < running_var->count(); ++c) {
    running_var_data[c] = Dtype(c + 1);
  }

  layer.Reshape(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  int num = this->blob_bottom_->num();
  int channels = this->blob_bottom_->channels();
  int height = this->blob_bottom_->height();
  int width = this->blob_bottom_->width();

  const Dtype kErrorBound = 0.001;
  for (int j = 0; j < channels; ++j) {
    for (int i = 0; i < num; ++i) {
      for ( int k = 0; k < height; ++k ) {
        for ( int l = 0; l < width; ++l ) {
          Dtype input = this->blob_bottom_->data_at(i, j, k, l);
          Dtype output = this->blob_top_->data_at(i, j, k, l);
          Dtype expect_output = (input - j) / sqrt(j + 1);
          EXPECT_NEAR(expect_output, output, kErrorBound);
        }
      }
    }
  }
}

TYPED_TEST(CuDNNBNLayerTest, TestBackwardFrozen) {
  typedef TypeParam Dtype;
  LayerParameter layer_param;

  BNParameter* bn_param = layer_param.mutable_bn_param();
  FillerParameter *slope_param = bn_param->mutable_slope_filler();
  slope_param->set_value(1);
  FillerParameter *bias_param = bn_param->mutable_bias_filler();
  bias_param->set_value(0);
  bn_param->set_eps(0.);
  bn_param->set_frozen(true);

  BNLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  // Init running mean and var
  shared_ptr<Blob<Dtype> > running_mean = layer.blobs()[2];
  Dtype* running_mean_data = running_mean->mutable_cpu_data();
  for (int c = 0; c < running_mean->count(); ++c) {
    running_mean_data[c] = Dtype(c);
  }
  shared_ptr<Blob<Dtype> > running_var = layer.blobs()[3];
  Dtype* running_var_data = running_var->mutable_cpu_data();
  for (int c = 0; c < running_var->count(); ++c) {
    running_var_data[c] = Dtype(c + 1);
  }

  layer.Reshape(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  Dtype* top_diff_data = this->blob_top_->mutable_cpu_diff();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    top_diff_data[i] = Dtype(i);
  }
  vector<bool> propagate_down(1, true);
  layer.Backward(this->blob_top_vec_, propagate_down, this->blob_bottom_vec_);

  int num = this->blob_bottom_->num();
  int channels = this->blob_bottom_->channels();
  int height = this->blob_bottom_->height();
  int width = this->blob_bottom_->width();

  const Dtype kErrorBound = 0.001;
  for (int j = 0; j < channels; ++j) {
    for (int i = 0; i < num; ++i) {
      for ( int k = 0; k < height; ++k ) {
        for ( int l = 0; l < width; ++l ) {
          Dtype input = this->blob_top_->diff_at(i, j, k, l);
          Dtype output = this->blob_bottom_->diff_at(i, j, k, l);
          Dtype expect_output = input / sqrt(j + 1);
          EXPECT_NEAR(expect_output, output, kErrorBound);
        }
      }
    }
  }
}
#endif

}

