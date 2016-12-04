#include <opencv2/core/core.hpp>

#include <string>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/highgui/highgui.hpp>

namespace caffe {

template<typename Dtype>
DataTransformer<Dtype>::DataTransformer(const TransformationParameter& param,
    Phase phase)
    : param_(param), phase_(phase) {
  // check if we want to use mean_file
  if (param_.has_mean_file()) {
    CHECK_EQ(param_.mean_value_size(), 0) <<
      "Cannot specify mean_file and mean_value at the same time";
    const string& mean_file = param.mean_file();
    LOG(INFO) << "Loading mean file from: " << mean_file;
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
    data_mean_.FromProto(blob_proto);
  }
  // check if we want to use mean_value
  if (param_.mean_value_size() > 0) {
    CHECK(param_.has_mean_file() == false) <<
      "Cannot specify mean_file and mean_value at the same time";
    for (int c = 0; c < param_.mean_value_size(); ++c) {
      mean_values_.push_back(param_.mean_value(c));
    }
  }

  // WuJiang, check if "label_offset" is used 
  if (param_.label_offset_size() > 0) {
    for (int c = 0; c < param_.label_offset_size(); ++c) {
      label_offset_.push_back(param_.label_offset(c));
      // LOG(INFO) << "label_offset_" << param_.label_offset(c);
    }
  }

  if (param_.label_scalefactor_size() > 0) {
    for (int c = 0; c < param_.label_scalefactor_size(); ++c) {
      label_scalefactor_.push_back(param_.label_scalefactor(c));
      //  LOG(INFO) << "label_scalefactor_" << param_.label_scalefactor(c);
    }
  }

}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const Datum& datum,
                                       Dtype* transformed_data) {
  const string& data = datum.data();
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();

  const int crop_size = param_.crop_size();
  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_uint8 = data.size() > 0;
  const bool has_mean_values = mean_values_.size() > 0;

  CHECK_GT(datum_channels, 0);
  CHECK_GE(datum_height, crop_size);
  CHECK_GE(datum_width, crop_size);

  Dtype* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(datum_channels, data_mean_.channels());
    CHECK_EQ(datum_height, data_mean_.height());
    CHECK_EQ(datum_width, data_mean_.width());
    mean = data_mean_.mutable_cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == datum_channels) <<
     "Specify either 1 mean_value or as many as channels: " << datum_channels;
    if (datum_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < datum_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }

  int height = datum_height;
  int width = datum_width;

  int h_off = 0;
  int w_off = 0;
  if (crop_size) {
    height = crop_size;
    width = crop_size;
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = Rand(datum_height - crop_size + 1);
      w_off = Rand(datum_width - crop_size + 1);
    } else {
      h_off = (datum_height - crop_size) / 2;
      w_off = (datum_width - crop_size) / 2;
    }
  }

  Dtype datum_element;
  int top_index, data_index;
  for (int c = 0; c < datum_channels; ++c) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        data_index = (c * datum_height + h_off + h) * datum_width + w_off + w;
        if (do_mirror) {
          top_index = (c * height + h) * width + (width - 1 - w);
        } else {
          top_index = (c * height + h) * width + w;
        }
        if (has_uint8) {
          datum_element =
            static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
        } else {
          datum_element = datum.float_data(data_index);
        }
        if (has_mean_file) {
          transformed_data[top_index] =
            (datum_element - mean[data_index]) * scale;
        } else {
          if (has_mean_values) {
            transformed_data[top_index] =
              (datum_element - mean_values_[c]) * scale;
          } else {
            transformed_data[top_index] = datum_element * scale;
          }
        }
      }
    }
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const Datum& datum,
                                       Blob<Dtype>* transformed_blob) {
  // If datum is encoded, decoded and transform the cv::image.
  if (datum.encoded()) {
    CHECK(!(param_.force_color() && param_.force_gray()))
        << "cannot set both force_color and force_gray";
    cv::Mat cv_img;
    if (param_.force_color() || param_.force_gray()) {
    // If force_color then decode in color otherwise decode in gray.
      cv_img = DecodeDatumToCVMat(datum, param_.force_color());
    } else {
      cv_img = DecodeDatumToCVMatNative(datum);
    }
    // Transform the cv::image into blob.
    return Transform(cv_img, transformed_blob);
  } else {
    if (param_.force_color() || param_.force_gray()) {
      LOG(ERROR) << "force_color and force_gray only for encoded datum";
    }
  }

  const int crop_size = param_.crop_size();
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();

  // Check dimensions.
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int num = transformed_blob->num();

  CHECK_EQ(channels, datum_channels);
  CHECK_LE(height, datum_height);
  CHECK_LE(width, datum_width);
  CHECK_GE(num, 1);

  if (crop_size) {
    CHECK_EQ(crop_size, height);
    CHECK_EQ(crop_size, width);
  } else {
    CHECK_EQ(datum_height, height);
    CHECK_EQ(datum_width, width);
  }

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();
  Transform(datum, transformed_data);
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const vector<Datum> & datum_vector,
                                       Blob<Dtype>* transformed_blob) {
  const int datum_num = datum_vector.size();
  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();

  CHECK_GT(datum_num, 0) << "There is no datum to add";
  CHECK_LE(datum_num, num) <<
    "The size of datum_vector must be no greater than transformed_blob->num()";
  Blob<Dtype> uni_blob(1, channels, height, width);
  for (int item_id = 0; item_id < datum_num; ++item_id) {
    int offset = transformed_blob->offset(item_id);
    uni_blob.set_cpu_data(transformed_blob->mutable_cpu_data() + offset);
    Transform(datum_vector[item_id], &uni_blob);
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const vector<cv::Mat> & mat_vector,
                                       Blob<Dtype>* transformed_blob) {
  const int mat_num = mat_vector.size();
  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();

  CHECK_GT(mat_num, 0) << "There is no MAT to add";
  CHECK_EQ(mat_num, num) <<
    "The size of mat_vector must be equals to transformed_blob->num()";
  Blob<Dtype> uni_blob(1, channels, height, width);
  for (int item_id = 0; item_id < mat_num; ++item_id) {
    int offset = transformed_blob->offset(item_id);
    uni_blob.set_cpu_data(transformed_blob->mutable_cpu_data() + offset);
    Transform(mat_vector[item_id], &uni_blob);
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const cv::Mat& cv_img,
                                       Blob<Dtype>* transformed_blob) {
  const int crop_size = param_.crop_size();
  const int img_channels = cv_img.channels();
  const int img_height = cv_img.rows;
  const int img_width = cv_img.cols;

  // Check dimensions.
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int num = transformed_blob->num();

  CHECK_EQ(channels, img_channels);
  CHECK_LE(height, img_height);
  CHECK_LE(width, img_width);
  CHECK_GE(num, 1);

  CHECK(cv_img.depth() == CV_8U || cv_img.depth() == CV_16U) << "Image data type must be unsigned byte";

  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;

  CHECK_GT(img_channels, 0);
  CHECK_GE(img_height, crop_size);
  CHECK_GE(img_width, crop_size);

  Dtype* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(img_channels, data_mean_.channels());
    CHECK_EQ(img_height, data_mean_.height());
    CHECK_EQ(img_width, data_mean_.width());
    mean = data_mean_.mutable_cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == img_channels) <<
     "Specify either 1 mean_value or as many as channels: " << img_channels;
    if (img_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < img_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }

  int h_off = 0;
  int w_off = 0;
  cv::Mat cv_cropped_img = cv_img;
  if (crop_size) {
    CHECK_EQ(crop_size, height);
    CHECK_EQ(crop_size, width);
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = Rand(img_height - crop_size + 1);
      w_off = Rand(img_width - crop_size + 1);
    } else {
      h_off = (img_height - crop_size) / 2;
      w_off = (img_width - crop_size) / 2;
    }
    cv::Rect roi(w_off, h_off, crop_size, crop_size);
    cv_cropped_img = cv_img(roi);
  } else {
    CHECK_EQ(img_height, height);
    CHECK_EQ(img_width, width);
  }

  CHECK(cv_cropped_img.data);

  cv_cropped_img.convertTo(cv_cropped_img, CV_16U);

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();
  int top_index;
  for (int h = 0; h < height; ++h) {
  //  const double* ptr = cv_cropped_img.ptr<double>(h);
    const uint16_t* ptr = cv_cropped_img.ptr<uint16_t>(h);
    int img_index = 0;
    for (int w = 0; w < width; ++w) {
      for (int c = 0; c < img_channels; ++c) {
        if (do_mirror) {
          top_index = (c * height + h) * width + (width - 1 - w);
        } else {
          top_index = (c * height + h) * width + w;
        }
        // int top_index = (c * height + h) * width + w;
        Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
        if (has_mean_file) {
          int mean_index = (c * img_height + h_off + h) * img_width + w_off + w;
          transformed_data[top_index] =
            (pixel - mean[mean_index]) * scale;
        } else {
          if (has_mean_values) {
            transformed_data[top_index] =
              (pixel - mean_values_[c]) * scale;
          } else {
            transformed_data[top_index] = pixel * scale;
          }
        }
      }
    }
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(Blob<Dtype>* input_blob,
                                       Blob<Dtype>* transformed_blob) {
  const int crop_size = param_.crop_size();
  const int input_num = input_blob->num();
  const int input_channels = input_blob->channels();
  const int input_height = input_blob->height();
  const int input_width = input_blob->width();

  if (transformed_blob->count() == 0) {
    // Initialize transformed_blob with the right shape.
    if (crop_size) {
      transformed_blob->Reshape(input_num, input_channels,
                                crop_size, crop_size);
    } else {
      transformed_blob->Reshape(input_num, input_channels,
                                input_height, input_width);
    }
  }

  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int size = transformed_blob->count();

  CHECK_LE(input_num, num);
  CHECK_EQ(input_channels, channels);
  CHECK_GE(input_height, height);
  CHECK_GE(input_width, width);


  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;

  int h_off = 0;
  int w_off = 0;
  if (crop_size) {
    CHECK_EQ(crop_size, height);
    CHECK_EQ(crop_size, width);
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = Rand(input_height - crop_size + 1);
      w_off = Rand(input_width - crop_size + 1);
    } else {
      h_off = (input_height - crop_size) / 2;
      w_off = (input_width - crop_size) / 2;
    }
  } else {
    CHECK_EQ(input_height, height);
    CHECK_EQ(input_width, width);
  }

  Dtype* input_data = input_blob->mutable_cpu_data();
  if (has_mean_file) {
    CHECK_EQ(input_channels, data_mean_.channels());
    CHECK_EQ(input_height, data_mean_.height());
    CHECK_EQ(input_width, data_mean_.width());
    for (int n = 0; n < input_num; ++n) {
      int offset = input_blob->offset(n);
      caffe_sub(data_mean_.count(), input_data + offset,
            data_mean_.cpu_data(), input_data + offset);
    }
  }

  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == input_channels) <<
     "Specify either 1 mean_value or as many as channels: " << input_channels;
    if (mean_values_.size() == 1) {
      caffe_add_scalar(input_blob->count(), -(mean_values_[0]), input_data);
    } else {
      for (int n = 0; n < input_num; ++n) {
        for (int c = 0; c < input_channels; ++c) {
          int offset = input_blob->offset(n, c);
          caffe_add_scalar(input_height * input_width, -(mean_values_[c]),
            input_data + offset);
        }
      }
    }
  }

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();

  for (int n = 0; n < input_num; ++n) {
    int top_index_n = n * channels;
    int data_index_n = n * channels;
    for (int c = 0; c < channels; ++c) {
      int top_index_c = (top_index_n + c) * height;
      int data_index_c = (data_index_n + c) * input_height + h_off;
      for (int h = 0; h < height; ++h) {
        int top_index_h = (top_index_c + h) * width;
        int data_index_h = (data_index_c + h) * input_width + w_off;
        if (do_mirror) {
          int top_index_w = top_index_h + width - 1;
          for (int w = 0; w < width; ++w) {
            transformed_data[top_index_w-w] = input_data[data_index_h + w];
          }
        } else {
          for (int w = 0; w < width; ++w) {
            transformed_data[top_index_h + w] = input_data[data_index_h + w];
          }
        }
      }
    }
  }
  if (scale != Dtype(1)) {
    DLOG(INFO) << "Scale: " << scale;
    caffe_scal(size, scale, transformed_data);
  }
}

template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(const Datum& datum) {
  if (datum.encoded()) {
    CHECK(!(param_.force_color() && param_.force_gray()))
        << "cannot set both force_color and force_gray";
    cv::Mat cv_img;
    if (param_.force_color() || param_.force_gray()) {
    // If force_color then decode in color otherwise decode in gray.
      cv_img = DecodeDatumToCVMat(datum, param_.force_color());
    } else {
      cv_img = DecodeDatumToCVMatNative(datum);
    }
    // InferBlobShape using the cv::image.
    return InferBlobShape(cv_img);
  }

  const int crop_size = param_.crop_size();
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();
  // Check dimensions.
  CHECK_GT(datum_channels, 0);
  CHECK_GE(datum_height, crop_size);
  CHECK_GE(datum_width, crop_size);
  // Build BlobShape.
  vector<int> shape(4);
  shape[0] = 1;
  shape[1] = datum_channels;
  shape[2] = (crop_size)? crop_size: datum_height;
  shape[3] = (crop_size)? crop_size: datum_width;
  return shape;
}

template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(
    const vector<Datum> & datum_vector) {
  const int num = datum_vector.size();
  CHECK_GT(num, 0) << "There is no datum to in the vector";
  // Use first datum in the vector to InferBlobShape.
  vector<int> shape = InferBlobShape(datum_vector[0]);
  // Adjust num to the size of the vector.
  shape[0] = num;
  return shape;
}

template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(const cv::Mat& cv_img) {
  const int crop_size = param_.crop_size();
  const int img_channels = cv_img.channels();
  const int img_height = cv_img.rows;
  const int img_width = cv_img.cols;
  // Check dimensions.
  CHECK_GT(img_channels, 0);
  CHECK_GE(img_height, crop_size);
  CHECK_GE(img_width, crop_size);
  // Build BlobShape.
  vector<int> shape(4);
  shape[0] = 1;
  shape[1] = img_channels;
  shape[2] = (crop_size)? crop_size: img_height;
  shape[3] = (crop_size)? crop_size: img_width;
  return shape;
}

template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(
    const vector<cv::Mat> & mat_vector) {
  const int num = mat_vector.size();
  CHECK_GT(num, 0) << "There is no cv_img to in the vector";
  // Use first cv_img in the vector to InferBlobShape.
  vector<int> shape = InferBlobShape(mat_vector[0]);
  // Adjust num to the size of the vector.
  shape[0] = num;
  return shape;
}

template <typename Dtype>
void DataTransformer<Dtype>::InitRand() {
  const bool needs_rand = param_.mirror() ||
      (phase_ == TRAIN && (param_.crop_size() || (param_.crop_h() && param_.crop_w())));
  if (needs_rand) {
    const unsigned int rng_seed = caffe_rng_rand();
    rng_.reset(new Caffe::RNG(rng_seed));
  } else {
    rng_.reset();
  }
}

template <typename Dtype>
int DataTransformer<Dtype>::Rand(int n) {
  CHECK(rng_);
  CHECK_GT(n, 0);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(rng_->generator());
  return ((*rng)() % n);
}

// add by shenli 0423/2016
// Rotation 
void Rotation(cv::Mat& src, int degree, bool islabel){
  int height = src.size().height;
  int width = src.size().width;

  cv::Point2f center = cv::Point2f(width / 2, height / 2);
  cv::Mat map_matrix = cv::getRotationMatrix2D(center, degree, 1.0);
  if (islabel){
    cv::warpAffine(src, src, map_matrix, src.size(), cv::INTER_NEAREST);
  } else{
    cv::warpAffine(src, src, map_matrix, src.size());
  }
}

// Aug 18,2016 by Chongruo: rotate 90,180,270 degrees ( 90 x n degrees ) closewise
// Rotate90n
void Rotate90n(cv::Mat& src, int n){
    CHECK_NEAR(n, 2, 1); // n >= 1, n<=3

    if (n==1){
        cv::transpose(src, src);
        cv::flip(src,src,1);
    }else if (n==2){
        cv::flip(src,src,-1);
    }else if (n==3){
        cv::transpose(src, src);
        cv::flip(src,src,0);
    }else{
        LOG(INFO) << " n should be int in [1,3] ";
    }
}

template<typename Dtype>
void DataTransformer<Dtype>::TransformImgAndSeg(const std::vector<cv::Mat>& cv_img_seg,
  Blob<Dtype>* transformed_data_blob, Blob<Dtype>* transformed_label_blob, const int ignore_label) {
  CHECK(cv_img_seg.size() == 2) << "Input must contain image and seg.";

  const int img_channels = cv_img_seg[0].channels();
  // height and width may change due to pad for cropping
  int img_height   = cv_img_seg[0].rows;
  int img_width    = cv_img_seg[0].cols;

  const int seg_channels = cv_img_seg[1].channels();
  int seg_height   = cv_img_seg[1].rows;
  int seg_width    = cv_img_seg[1].cols;

  const int data_channels = transformed_data_blob->channels();
  const int data_height   = transformed_data_blob->height();
  const int data_width    = transformed_data_blob->width();

  const int label_channels = transformed_label_blob->channels();
  const int label_height   = transformed_label_blob->height();
  const int label_width    = transformed_label_blob->width();

  CHECK_EQ(seg_channels, 1);
  CHECK_EQ(img_channels, data_channels);
  CHECK_EQ(img_height, seg_height);
  CHECK_EQ(img_width, seg_width);

  CHECK_EQ(label_channels, 1);
  CHECK_EQ(data_height, label_height);
  CHECK_EQ(data_width, label_width);

  CHECK(cv_img_seg[0].depth() == CV_8U || cv_img_seg[0].depth() == CV_16U ) << "Image data type must be unsigned byte";
  CHECK(cv_img_seg[1].depth() == CV_8U) << "Seg data type must be unsigned byte";

  const int crop_size = param_.crop_size();
  int crop_w = param_.crop_w();
  int crop_h = param_.crop_h();
  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;

  CHECK_GT(img_channels, 0);

  Dtype* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(img_channels, data_mean_.channels());
    CHECK_EQ(img_height, data_mean_.height());
    CHECK_EQ(img_width, data_mean_.width());
    mean = data_mean_.mutable_cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == img_channels) <<
     "Specify either 1 mean_value or as many as channels: " << img_channels;
    if (img_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < img_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }
  int h_off = 0;
  int w_off = 0;
  cv::Mat cv_cropped_img = cv_img_seg[0];  
  cv::Mat cv_cropped_seg = cv_img_seg[1];
  
  // transform to double, since we will pad mean pixel values
  cv_cropped_img.convertTo(cv_cropped_img, CV_64F);

  // add rotation augmentation by shenli 0423/2016
  if (param_.rand_rotate_size() > 0){
    CHECK_EQ(this->param_.rand_rotate_size(), 2) << "Exactly two rand_rotate param required";
    const float rand_rotate_small = param_.rand_rotate(0);
    const float rand_rotate_large = param_.rand_rotate(1);
    CHECK_LT(rand_rotate_small, rand_rotate_large) << "first rand_ratate should be smaller than the second rand_rotate";
    cv::Mat cv_aug_img = cv_cropped_img;
    cv::Mat cv_aug_seg = cv_cropped_seg;
    int rand_value = Rand(2);
    if (rand_value == 1){
      const float angl = rand_rotate_small + (rand_rotate_large - rand_rotate_small) * (caffe_rng_rand() % 101l) / 100.0f;
      Rotation(cv_aug_img, angl, 0);
      Rotation(cv_aug_seg, angl, 1);
    }
    cv_cropped_img = cv_aug_img;
    cv_cropped_seg = cv_aug_seg;
  }


  if (param_.gaussian_blur() && Rand(2)) {
    double rand_sigma = (rand() % 100) / 100.0 * 0.6;
    cv::GaussianBlur(cv_cropped_img, cv_cropped_img, cv::Size( 5, 5 ), rand_sigma, rand_sigma);
  }

  // Check if we need to pad img to fit for crop_size
  // copymakeborder
  if (crop_size > 0)
  {
    crop_h = crop_size;
    crop_w = crop_size;
  }
  int pad_height = std::max(crop_h - img_height, 0);
  int pad_width = std::max(crop_w - img_width, 0);
  int pad_h_half = pad_height/2;
  int pad_w_half = pad_width/2;
  if (pad_height > 0 || pad_width > 0) {
    cv::copyMakeBorder(cv_cropped_img, cv_cropped_img, pad_h_half, pad_height-pad_h_half, 
          pad_w_half, pad_width-pad_w_half, cv::BORDER_CONSTANT, 
		       cv::Scalar(float(mean_values_[0]), float(mean_values_[1]), float(mean_values_[2])));
    cv::copyMakeBorder(cv_cropped_seg, cv_cropped_seg, pad_h_half, pad_height-pad_h_half, 
          pad_w_half, pad_width-pad_w_half, cv::BORDER_CONSTANT, 
           cv::Scalar(ignore_label));
    // update height/width
    img_height   = cv_cropped_img.rows;
    img_width    = cv_cropped_img.cols;

    seg_height   = cv_cropped_seg.rows;
    seg_width    = cv_cropped_seg.cols;
  }
   

  // crop img/seg
  if (crop_h && crop_w) {
    CHECK_EQ(crop_h, data_height);
    CHECK_EQ(crop_w, data_width);    
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = Rand(img_height - crop_h + 1);
      w_off = Rand(img_width - crop_w + 1);
    } else {
      // CHECK: use middle crop
      h_off = (img_height - crop_h) / 2;
      w_off = (img_width - crop_w) / 2;
    }
    cv::Rect roi(w_off, h_off, crop_w, crop_h);
    cv_cropped_img = cv_cropped_img(roi);
    cv_cropped_seg = cv_cropped_seg(roi);
  } 
  
  CHECK(cv_cropped_img.data);
  CHECK(cv_cropped_seg.data);


  // By Chongruo, Aug 18,2016:  rotate 90 x n degrees clockwise
  // Moved by WuJiang
  if ( param_.rotate90n() ){
    int rand_value = Rand(4);
    if (rand_value > 0){
        Rotate90n(cv_cropped_img, rand_value);
        Rotate90n(cv_cropped_seg, rand_value);
    }
  }


  Dtype* transformed_data  = transformed_data_blob->mutable_cpu_data();
  Dtype* transformed_label = transformed_label_blob->mutable_cpu_data();

  int top_index;
  const double* data_ptr;
  const uchar* label_ptr;

  for (int h = 0; h < data_height; ++h) {
    data_ptr = cv_cropped_img.ptr<double>(h);
    label_ptr = cv_cropped_seg.ptr<uchar>(h);

    int data_index = 0;
    int label_index = 0;

    for (int w = 0; w < data_width; ++w) {
      // for image
      for (int c = 0; c < img_channels; ++c) {
        if (do_mirror) {
          top_index = (c * data_height + h) * data_width + (data_width - 1 - w);
        } else {
          top_index = (c * data_height + h) * data_width + w;
        }
        Dtype pixel = static_cast<Dtype>(data_ptr[data_index++]);
        if (has_mean_file) {
          int mean_index = (c * img_height + h_off + h) * img_width + w_off + w;
          transformed_data[top_index] =
            (pixel - mean[mean_index]) * scale;
        } else {
          if (has_mean_values) {
            transformed_data[top_index] =
              (pixel - mean_values_[c]) * scale;
          } else {
            transformed_data[top_index] = pixel * scale;
          }
        }
      }

      // for segmentation
      if (do_mirror) {
        top_index = h * data_width + data_width - 1 - w;
      } else {
        top_index = h * data_width + w;
      }
      Dtype pixel = static_cast<Dtype>(label_ptr[label_index++]);
      transformed_label[top_index] = pixel;
    }
  }
}

// Add WuJiang 20160612
template<typename Dtype>
void DataTransformer<Dtype>::TransformImgAndSeg(const std::vector<cv::Mat>& cv_img_seg,
  Blob<Dtype>* transformed_data_blob, std::vector<Blob<Dtype>*> transformed_label_blob_vector, const int ignore_label)
{

  // 1. Init Checking 
  CHECK(cv_img_seg.size() >= 2) << "Input must contain image and at least one seg.";  
  CHECK(cv_img_seg.size() - 1 == transformed_label_blob_vector.size()) << "Input label number == " << cv_img_seg.size() - 1 <<
    "~= Output label number ==" << transformed_label_blob_vector.size();

  const int label_num = transformed_label_blob_vector.size();

  // height and width may change due to pad for cropping
  const int img_channels = cv_img_seg[0].channels();
  int img_height = cv_img_seg[0].rows;
  int img_width = cv_img_seg[0].cols;

  const int data_channels = transformed_data_blob->channels();
  const int data_height   = transformed_data_blob->height();
  const int data_width    = transformed_data_blob->width();

  CHECK_GT(img_channels, 0);
  CHECK_EQ(img_channels, data_channels);
  CHECK(cv_img_seg[0].depth() == CV_8U || cv_img_seg[0].depth() == CV_16U ) << "Image data type must be unsigned byte";

  int seg_channels,seg_height, seg_width;
  int label_channels,label_height, label_width;
  for (int label_index = 0; label_index < label_num; label_index++){
    seg_channels = cv_img_seg[label_index+1].channels();
    seg_height   = cv_img_seg[label_index+1].rows;
    seg_width    = cv_img_seg[label_index+1].cols;

    label_channels = transformed_label_blob_vector[label_index]->channels();
    label_height   = transformed_label_blob_vector[label_index]->height();
    label_width    = transformed_label_blob_vector[label_index]->width();

    CHECK_EQ(img_height, seg_height);
    CHECK_EQ(img_width , seg_width);
    CHECK_EQ(data_height, label_height);
    CHECK_EQ(data_width , label_width);
    CHECK_EQ(seg_channels, label_channels);
    CHECK_EQ(label_channels, 1);

    CHECK(cv_img_seg[1 + label_index].depth() == CV_8U) \
         << " The " << label_index << "th Seg " << "type must be unsigned byte";
  }




  // 2. Preprocessing
  const int crop_size = param_.crop_size();
  int crop_h = param_.crop_h();
  int crop_w = param_.crop_w();
  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;
  const bool has_label_offset = label_offset_.size() > 0;
  const bool has_label_scale_ = label_scalefactor_.size() > 0;

  if (has_label_offset)
    CHECK_EQ(label_offset_.size(), label_num) << "label_offset_.size() ~= label_num";
  if (has_label_scale_)
    CHECK_EQ(label_scalefactor_.size(), label_num) << "label_scalefactor_.size() ~= label_num";


  Dtype* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(img_channels, data_mean_.channels());
    CHECK_EQ(img_height, data_mean_.height());
    CHECK_EQ(img_width, data_mean_.width());
    mean = data_mean_.mutable_cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == img_channels) <<
        "Specify either 1 mean_value or as many as channels: " << img_channels;
    if (img_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicityINSTANTIATE_CLASS(DataTransformer);
      for (int c = 1; c < img_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }

  cv::Mat cv_cropped_img = cv_img_seg[0];
  std::vector<cv::Mat> cv_cropped_seg_vector;
  for (int i_label = 0; i_label < label_num; i_label++)
    cv_cropped_seg_vector.push_back(cv_img_seg[1 + i_label]);


  // the first operation should be the one that can change the image size
  // if that, cv_cropped_img will be an entity, not a pointer.  Origianl data(cv_img_seg) won't be changed 

  // 2.1 random scale, Jianping 2016.06.08
  if (param_.rand_resize_size() > 0 && phase_ == TRAIN) {
     CHECK_EQ(param_.rand_resize_size(), 2) << "Exactly two rand_resize param required";
     const float rand_resize_small = param_.rand_resize(0);
     const float rand_resize_large = param_.rand_resize(1);
     CHECK_LT(rand_resize_small, rand_resize_large) << "first rand_resize should be smaller than the second rand_resize";
     const float temp_scale = rand_resize_small + (rand_resize_large - rand_resize_small) * Rand(101) / 100.0f;
     resize(cv_cropped_img, cv_cropped_img, cv::Size(0, 0), temp_scale, temp_scale);
     for (int label_index = 0; label_index < label_num; label_index++)
       resize(cv_cropped_seg_vector[label_index], cv_cropped_seg_vector[label_index], cv::Size(0, 0), temp_scale, temp_scale, CV_INTER_NN);


     // update height/width
     img_height = cv_cropped_img.rows;
     img_width = cv_cropped_img.cols;
     seg_height = cv_cropped_seg_vector[0].rows;
     seg_width  = cv_cropped_seg_vector[0].cols;
   }




  // 2.2 random rotate,  add augmentation 0423/2016
  if (param_.rand_rotate_size() > 0){
    CHECK_EQ(this->param_.rand_rotate_size(), 2) << "Exactly two rand_rotate param required";
    const float rand_rotate_small = param_.rand_rotate(0);
    const float rand_rotate_large = param_.rand_rotate(1);
    CHECK_LT(rand_rotate_small, rand_rotate_large) \
              << "first rand_ratate should be smaller than the second rand_rotate";

    int rand_value = Rand(2);
    if (rand_value == 1){
      const float angl = rand_rotate_small + (rand_rotate_large - rand_rotate_small) * (caffe_rng_rand() % 101l) / 100.0f;
      Rotation(cv_cropped_img, angl, 0);
      for (int label_index = 0; label_index < label_num; label_index++)
        Rotation((cv_cropped_seg_vector[label_index]), angl, 1);
    }
  }





  // 2.3. gaussian_blur
  if (param_.gaussian_blur() && Rand(2)) {
    double rand_sigma = (rand() % 100) / 100.0 * 0.6;
    cv::GaussianBlur(cv_cropped_img, cv_cropped_img, cv::Size( 5, 5 ), rand_sigma, rand_sigma);
    //for (int label_index = 0; label_index < label_num; label_index++)
        //cv::GaussianBlur(cv_cropped_seg_vector[label_index], cv_cropped_seg_vector[label_index], cv::Size( 5, 5 ), rand_sigma, rand_sigma);
  }

  // 2.4. crop 
  // transform to double, since we will pad mean pixel values
  cv_cropped_img.convertTo(cv_cropped_img, CV_64F);

  // Check if we need to pad img to fit for crop_size
  // copymakeborder
  if (crop_size > 0)
  {
    crop_h = crop_size;
    crop_w = crop_size;
  }
  int pad_height = std::max(crop_h - img_height, 0);
  int pad_width = std::max(crop_w - img_width, 0);
  int pad_h_half = pad_height/2;
  int pad_w_half = pad_width/2;
  if (pad_height > 0 || pad_width > 0) {
    cv::copyMakeBorder(cv_cropped_img, cv_cropped_img, pad_h_half, pad_height-pad_h_half,
        pad_w_half, pad_width-pad_w_half, cv::BORDER_CONSTANT,
    cv::Scalar((float)mean_values_[0], (float)mean_values_[1], (float)mean_values_[2]));
    for (int label_index = 0; label_index < label_num; label_index++)
      cv::copyMakeBorder(cv_cropped_seg_vector[label_index], cv_cropped_seg_vector[label_index], 
          pad_h_half, pad_height-pad_h_half, pad_w_half, pad_width-pad_w_half, cv::BORDER_CONSTANT,cv::Scalar(ignore_label));

    // update height/width
    img_height = cv_cropped_img.rows;
    img_width  = cv_cropped_img.cols;
    seg_height = cv_cropped_seg_vector[0].rows;
    seg_width  = cv_cropped_seg_vector[0].cols;
  }


  int h_off = 0;
  int w_off = 0;
  // crop img/seg	  
  if (crop_h || crop_w) {
    CHECK_EQ(crop_h, data_height);
    CHECK_EQ(crop_w, data_width);
	  // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = Rand(img_height - crop_h + 1);
      w_off = Rand(img_width - crop_w + 1);
    }
    else {
      // CHECK: use middle crop
      h_off = (img_height - crop_h) / 2;
      w_off = (img_width - crop_w) / 2;
    }
    cv::Rect roi(w_off, h_off, crop_w, crop_h);
    cv_cropped_img = cv_cropped_img(roi);
    for (int i_label = 0; i_label < label_num; i_label++)
      cv_cropped_seg_vector[i_label] = cv_cropped_seg_vector[i_label](roi);
  }

  CHECK(cv_cropped_img.data);
  for (int i_label = 0; i_label < label_num; i_label++)
    CHECK(cv_cropped_seg_vector[i_label].data);
  

  // 2.5. rotate90n,  By Chongruo, Aug 18,2016:  rotate 90 x n degrees clockwise
  // Moved by WuJiang
  if (param_.rotate90n()){
	  int rand_value = Rand(4);
	  if (rand_value > 0){
		  Rotate90n(cv_cropped_img, rand_value);
		  for (int label_index = 0; label_index < label_num; label_index++)
			  Rotate90n((cv_cropped_seg_vector[label_index]), rand_value);
	  }
  }



  // 2.6 mean, mirror
  Dtype* transformed_data = transformed_data_blob->mutable_cpu_data();
  std::vector<Dtype*> transformed_label_vector(label_num);
  for (int label_index = 0; label_index < label_num; label_index++)
    transformed_label_vector[label_index] = transformed_label_blob_vector[label_index]->mutable_cpu_data();
  
  int top_index;
  const double* data_ptr;
  std::vector< const uchar* > label_ptr_vector(label_num);
  
  for (int h = 0; h < data_height; ++h) {
    data_ptr = cv_cropped_img.ptr<double>(h);

    for (int label_index = 0; label_index < label_num;label_index++)
      label_ptr_vector[label_index] = cv_cropped_seg_vector[label_index].ptr<uchar>(h);
	  
    int data_index = 0;
    int label_index_g = 0;
  
    for (int w = 0; w < data_width; ++w) {
      // for image
      for (int c = 0; c < img_channels; ++c) {
        if (do_mirror) {
          top_index = (c * data_height + h) * data_width + (data_width - 1 - w);
        } else {
          top_index = (c * data_height + h) * data_width + w;
        }
        Dtype pixel = static_cast<Dtype>(data_ptr[data_index++]);
        if (has_mean_file) {
          int mean_index = (c * img_height + h_off + h) * img_width + w_off + w;
          transformed_data[top_index] =
          (pixel - mean[mean_index]) * scale;
        }
        else {
          if (has_mean_values) {
            transformed_data[top_index] = (pixel - mean_values_[c]) * scale;
          }
          else {
            transformed_data[top_index] = pixel * scale;
          }
        }
      }

      // for segmentation
      if (do_mirror) {
        top_index = h * data_width + data_width - 1 - w;
      } else {
        top_index = h * data_width + w;
      }
      for (int label_index = 0; label_index < label_num; label_index++) {
        Dtype pixel = static_cast<Dtype>(label_ptr_vector[label_index][label_index_g]);
        if (has_label_scale_)
          pixel = pixel * label_scalefactor_[label_index];
        if (has_label_offset)
          pixel = pixel + label_offset_[label_index];

        transformed_label_vector[label_index][top_index] = pixel;
      }
      label_index_g++;
    }
  }
} 

INSTANTIATE_CLASS(DataTransformer);
 
}  // namespace caffe
