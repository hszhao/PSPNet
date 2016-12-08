#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/mat_read_layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/matio_io.hpp"
#include <sstream>

namespace caffe {

template <typename Dtype>
void MatReadLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
				      const vector<Blob<Dtype>*>& top) {
  prefix_ = this->layer_param_.mat_read_param().prefix();
  batch_size_ = this->layer_param_.mat_read_param().batch_size();
  CHECK_GT(batch_size_, 0) << "batch_size must be positive";
  reshape_ = false;
  iter_ = 0;
  if (this->layer_param_.mat_read_param().has_source()) {
    std::ifstream infile(this->layer_param_.mat_read_param().source().c_str());
    CHECK(infile.good()) << "Failed to open source file "
			 << this->layer_param_.mat_read_param().source();
    const int strip = this->layer_param_.mat_read_param().strip();
    CHECK_GE(strip, 0) << "Strip cannot be negative";
    string linestr;
    while (std::getline(infile, linestr)) {
      std::istringstream iss(linestr);
      string filename;
      iss >> filename;
      CHECK_GT(filename.size(), strip) << "Too much stripping";
      fnames_.push_back(filename.substr(0, filename.size() - strip));
    }
    LOG(INFO) << "MatRead will load from a set of " << fnames_.size() << " files.";
    CHECK_GT(fnames_.size(), 0);
  }

  // Read an input, and use it to initialize the top blob.
  for (int i = 0; i < top.size(); ++i) {
    std::ostringstream oss;
    oss << prefix_;
    oss << fnames_[0];
    oss << "_blob_" << i << ".mat";
    Blob<Dtype> blob;
    ReadBlobFromMat(oss.str().c_str(), &blob);
    CHECK_EQ(blob.num(), 1);
    LOG(INFO) << "Matread setup top info: " << batch_size_ << "," <<  blob.channels() << "," << blob.height() << "," << blob.width();
    top[i]->Reshape(batch_size_, blob.channels(), blob.height(), blob.width());
    
  }
}

template <typename Dtype>
void MatReadLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // reshape_ = true;  // iter_ = 0;
  //Forward_cpu(bottom, top);
  //reshape_ = false; // iter_ = 0;
}

template <typename Dtype>
void MatReadLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  for (int n = 0; n < batch_size_; ++n) {
    for (int i = 0; i < top.size(); ++i) {
      std::ostringstream oss;
      oss << prefix_;
      if (this->layer_param_.mat_read_param().has_source()) {
	if (iter_ >= fnames_.size()) {
	  iter_ = 0;
	}
	oss << fnames_[iter_];
      }
      else {
	oss << "iter_" << iter_;
      }
      oss << "_blob_" << i << ".mat";
      Blob<Dtype> blob;
      ReadBlobFromMat(oss.str().c_str(), &blob);
      CHECK_EQ(blob.num(), 1);
      if (reshape_) {
	top[i]->Reshape(batch_size_, blob.channels(), blob.height(), blob.width());
      }
      else {
	CHECK(blob.channels()  == top[i]->channels()
	      && blob.height() == top[i]->height() 
	      && blob.width()  == top[i]->width());
      }
      caffe_copy(blob.count(), blob.cpu_data(),
	  top[i]->mutable_cpu_data_at(n));
    }
    ++iter_;
  }
}

template <typename Dtype>
void MatReadLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  return;
}

INSTANTIATE_CLASS(MatReadLayer);
REGISTER_LAYER_CLASS(MatRead);

}  // namespace caffe
