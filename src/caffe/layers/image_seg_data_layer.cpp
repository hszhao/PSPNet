#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/data_transformer.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
ImageSegDataLayer<Dtype>::~ImageSegDataLayer<Dtype>() {
  this->JoinPrefetchThread();
}

template <typename Dtype>
void ImageSegDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.image_data_param().new_height();
  const int new_width  = this->layer_param_.image_data_param().new_width();
  const bool is_color  = this->layer_param_.image_data_param().is_color();
  const int label_type = this->layer_param_.image_data_param().label_type();

  const bool is_rsdata  = this->layer_param_.image_data_param().is_rsdata();

  const bool is_modis_bin = this->layer_param_.image_data_param().is_modis_bin();
  if(is_modis_bin){
    CHECK(this->layer_param_.image_data_param().modis_class_num() == 4 || this->layer_param_.image_data_param().modis_class_num() == 2) << "modis_class_num must be 2 or 4, default = 4.";
    CHECK(label_type == ImageDataParameter_LabelType_PIXEL) << "label type of MODIS label must be PIXEL";
    if(this->layer_param_.image_data_param().has_modis_channles()){
      int modis_channles_ = this->layer_param_.image_data_param().modis_channles();
      CHECK_GT(modis_channles_,0) << "modis_channles should be greater than 0!";
      CHECK_EQ(this->layer_param_.image_data_param().modis_channel_list().size(), 0) << "modis_channels | modis_channel_list required!";
      channel_list_.resize(modis_channles_);
      for(int i = 0; i < modis_channles_; i++){
        channel_list_[i] = i;
      }
    }
    else{
      int modis_channles_ = this->layer_param_.image_data_param().modis_channel_list().size();
      CHECK_GT(modis_channles_,0) << "length(modis_channel_list) > 0!";
      channel_list_.resize(modis_channles_);
      for(int i = 0; i < modis_channles_; i++){
        channel_list_[i] = this->layer_param_.image_data_param().modis_channel_list(i);
      }
      sort(channel_list_.begin(),channel_list_.end());
    }
  }


  string root_folder = this->layer_param_.image_data_param().root_folder();

  TransformationParameter transform_param = this->layer_param_.transform_param();
  CHECK(transform_param.has_mean_file() == false) << 
         "ImageSegDataLayer does not support mean file";
  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";

  // Read the file with filenames and labels
  const string& source = this->layer_param_.image_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());

  string linestr;
  while (std::getline(infile, linestr)) {
    std::istringstream iss(linestr);
    string imgfn;
    iss >> imgfn;
    string segfn = "";
    if (label_type != ImageDataParameter_LabelType_NONE) {
      iss >> segfn;
    }
    lines_.push_back(std::make_pair(imgfn, segfn));
  }

  if (this->layer_param_.image_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";

  const int thread_id = Caffe::getThreadId();
  int thread_num = Caffe::getThreadNum();
  if (thread_num == 0){
    thread_num = 1;
  }
  lines_id_ = lines_.size() / thread_num * thread_id;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.image_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.image_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size() / thread_num, skip) << "Not enough points to skip";
    lines_id_ += skip;
  }

  // Read an image, and use it to initialize the top blob.
  cv::Mat cv_img;

  bool is_label = false;
  if(!is_modis_bin){
    if (is_rsdata) {
      cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first, new_height, new_width, is_color, is_label, is_rsdata);
    } else {
      cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first, new_height, new_width, is_color, is_label, false);
    }
  }
  int channels = cv_img.channels();
  const int height = cv_img.rows;
  const int width = cv_img.cols;
  // image
  const int crop_size = this->layer_param_.transform_param().crop_size();
  int crop_h = this->layer_param_.transform_param().crop_h();
  int crop_w = this->layer_param_.transform_param().crop_w();
std::cout << "aaa" << this->layer_param_.image_data_param().batch_size() << "bbb" << thread_num << std::endl;
  const int batch_size = this->layer_param_.image_data_param().batch_size() / thread_num;
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  CHECK(!(crop_size > 0 && (crop_h > 0 || crop_w > 0)))
	  << "Use either crop_size or crop_h & crop_w, not both.";
  CHECK((crop_h > 0 && crop_w > 0) || (crop_h == 0 && crop_w == 0))
	  << "crop_h and crop_w should be used at the same time and should be positive.";

  if(is_modis_bin){
    CHECK(crop_size > 0 || (crop_h > 0 && crop_w > 0)) << "crop_size > 0 if MODIS bin data, no new_height and new_width";
    channels = channel_list_.size();
  }
  if (crop_size > 0 || (crop_h > 0 && crop_w > 0)) {
    if (crop_size > 0)
    {
      crop_h = crop_size;
      crop_w = crop_size;
    }
    top[0]->Reshape(batch_size, channels, crop_h, crop_w);
    this->prefetch_data_.Reshape(batch_size, channels, crop_h, crop_w);
    this->transformed_data_.Reshape(1, channels, crop_h, crop_w);

    //label
    top[1]->Reshape(batch_size, 1, crop_h, crop_w);
    this->prefetch_label_.Reshape(batch_size, 1, crop_h, crop_w);
    this->transformed_label_.Reshape(1, 1, crop_h, crop_w);
     
  } else {
    top[0]->Reshape(batch_size, channels, height, width);
    this->prefetch_data_.Reshape(batch_size, channels, height, width);
    this->transformed_data_.Reshape(1, channels, height, width);

    //label
    top[1]->Reshape(batch_size, 1, height, width);
    this->prefetch_label_.Reshape(batch_size, 1, height, width);
    this->transformed_label_.Reshape(1, 1, height, width);     
  }

  // image dimensions, for each image, stores (img_height, img_width)
  top[2]->Reshape(batch_size, 1, 1, 2);
  this->prefetch_data_dim_.Reshape(batch_size, 1, 1, 2);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
	    << top[0]->channels() << "," << top[0]->height() << ","
	    << top[0]->width();
  // label
  LOG(INFO) << "output label size: " << top[1]->num() << ","
	    << top[1]->channels() << "," << top[1]->height() << ","
	    << top[1]->width();
  // image_dim
  LOG(INFO) << "output data_dim size: " << top[2]->num() << ","
	    << top[2]->channels() << "," << top[2]->height() << ","
	    << top[2]->width();
}

template <typename Dtype>
void ImageSegDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void ImageSegDataLayer<Dtype>::InternalThreadEntry() {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(this->prefetch_data_.count());
  CHECK(this->transformed_data_.count());

  Dtype* top_data     = this->prefetch_data_.mutable_cpu_data();
  Dtype* top_label    = this->prefetch_label_.mutable_cpu_data(); 
  Dtype* top_data_dim = this->prefetch_data_dim_.mutable_cpu_data();

  const int max_height = this->prefetch_data_.height();
  const int max_width  = this->prefetch_data_.width();

  ImageDataParameter image_data_param    = this->layer_param_.image_data_param();
  int thread_num = Caffe::getThreadNum();
  if (thread_num == 0){
    thread_num = 1;
  }
  const int batch_size = image_data_param.batch_size() / thread_num;
  const int new_height = image_data_param.new_height();
  const int new_width  = image_data_param.new_width();
  const int label_type = this->layer_param_.image_data_param().label_type();
  const int ignore_label = image_data_param.ignore_label();
  const bool is_color  = image_data_param.is_color();
  const bool is_rsdata = image_data_param.is_rsdata();
  const bool is_modis_bin = image_data_param.is_modis_bin();
  string root_folder   = image_data_param.root_folder();

  const int lines_size = lines_.size();
  int top_data_dim_offset;

  for (int item_id = 0; item_id < batch_size; ++item_id) {
    top_data_dim_offset = this->prefetch_data_dim_.offset(item_id);

    std::vector<cv::Mat> cv_img_seg;

    // get a blob
    timer.Start();
    CHECK_GT(lines_size, lines_id_);

    int img_row, img_col;
    bool is_label = false;

    if(is_modis_bin){
      cv_img_seg.push_back(ReadImageToCVMat(root_folder + lines_[lines_id_].first,
          new_height, new_width, is_color, is_label, is_rsdata, is_modis_bin, channel_list_, &img_row, &img_col));
    }
    else{
      if (is_rsdata) {
        cv_img_seg.push_back(ReadImageToCVMat(root_folder + lines_[lines_id_].first,
          new_height, new_width, is_color, is_label, is_rsdata, &img_row, &img_col));
      } else {
        cv_img_seg.push_back(ReadImageToCVMat(root_folder + lines_[lines_id_].first,
          new_height, new_width, is_color, is_label, &img_row, &img_col));
      }
    }

    top_data_dim[top_data_dim_offset]     = static_cast<Dtype>(std::min(max_height, img_row));
    top_data_dim[top_data_dim_offset + 1] = static_cast<Dtype>(std::min(max_width, img_col));

    if (!cv_img_seg[0].data) {
      DLOG(INFO) << "Fail to load img: " << root_folder + lines_[lines_id_].first;
    }
    if (label_type == ImageDataParameter_LabelType_PIXEL) {
      is_label = true;
      if(is_modis_bin){
        cv::Mat labelmat = ReadImageToCVMat(root_folder + lines_[lines_id_].second,
              new_height, new_width, false, is_label, false, is_modis_bin);
        if(this->layer_param_.image_data_param().modis_class_num() == 2){
          labelmat = (labelmat >= 2)/255;
        }
        cv_img_seg.push_back(labelmat);
        CHECK_EQ(cv_img_seg[0].rows,cv_img_seg[1].rows);
        CHECK_EQ(cv_img_seg[0].cols,cv_img_seg[1].cols);

      }else{
        cv_img_seg.push_back(ReadImageToCVMat(root_folder + lines_[lines_id_].second,
              new_height, new_width, false, is_label, false));
      }
      if (!cv_img_seg[1].data) {
        DLOG(INFO) << "Fail to load seg: " << root_folder + lines_[lines_id_].second;
      }
    }
    else if (label_type == ImageDataParameter_LabelType_IMAGE) {
      const int label = atoi(lines_[lines_id_].second.c_str());
      cv::Mat seg(cv_img_seg[0].rows, cv_img_seg[0].cols, 
		  CV_8UC1, cv::Scalar(label));
      cv_img_seg.push_back(seg);      
    }
    else {
      cv::Mat seg(cv_img_seg[0].rows, cv_img_seg[0].cols, 
		  CV_8UC1, cv::Scalar(ignore_label));
      cv_img_seg.push_back(seg);
    }

    // Apply random scale, Jianping 2016.06.08
    if (this->layer_param_.transform_param().rand_resize_size() > 0) {
      CHECK_EQ(this->layer_param_.transform_param().rand_resize_size(), 2) << "Exactly two rand_resize param required";
      const float rand_resize_small = this->layer_param_.transform_param().rand_resize(0);
      const float rand_resize_large = this->layer_param_.transform_param().rand_resize(1);
      CHECK_LT(rand_resize_small, rand_resize_large) << "first rand_resize should be smaller than the second rand_resize";
      const float temp_scale = rand_resize_small + (rand_resize_large - rand_resize_small) * (caffe_rng_rand() % 101l) / 100.0f;

      float aspect_ratio = 1.0f;
      if (this->layer_param_.transform_param().rand_aspect_ratio_size() > 0 && (rand() % 2)) {
        CHECK_EQ(this->layer_param_.transform_param().rand_aspect_ratio_size(), 2) << "Exactly two rand_aspect_ratio param required";
        const float rand_aspect_ratio_small = this->layer_param_.transform_param().rand_aspect_ratio(0);
        const float rand_aspect_ratio_large = this->layer_param_.transform_param().rand_aspect_ratio(1);
        CHECK_LT(rand_aspect_ratio_small, rand_aspect_ratio_large) << "first rand_resize should be smaller than the second rand_resize";
        aspect_ratio = rand_aspect_ratio_small + (rand_aspect_ratio_large - rand_aspect_ratio_small) * (caffe_rng_rand() % 101l) / 100.0f;
        aspect_ratio = sqrt(aspect_ratio);
      }
      const float scale_factor_x = temp_scale * aspect_ratio;
      const float scale_factor_y = temp_scale / aspect_ratio;
      resize(cv_img_seg[0], cv_img_seg[0], cv::Size(0, 0), scale_factor_x, scale_factor_y);
      resize(cv_img_seg[1], cv_img_seg[1], cv::Size(0, 0), scale_factor_x, scale_factor_y, CV_INTER_NN);
    }

    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply transformations (mirror, crop...) to the image
    int offset;

    offset = this->prefetch_data_.offset(item_id);
    this->transformed_data_.set_cpu_data(top_data + offset);

    offset = this->prefetch_label_.offset(item_id);
    this->transformed_label_.set_cpu_data(top_label + offset);

    this->data_transformer_->TransformImgAndSeg(cv_img_seg, 
    	 &(this->transformed_data_), &(this->transformed_label_),
    	 ignore_label);
    trans_time += timer.MicroSeconds();

    // go to the next std::vector<int>::iterator iter;
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.image_data_param().shuffle()) {
        ShuffleImages();
      }
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(ImageSegDataLayer);
REGISTER_LAYER_CLASS(ImageSegData);

}  // namespace caffe
