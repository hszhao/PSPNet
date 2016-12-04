#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/mpi_functions.hpp"

namespace caffe {

template <typename Dtype>
void ScatterLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom.size(), top.size())
      << "The number of bottom and top blobs must be the same";
}

template <typename Dtype>
void ScatterLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
#ifdef USE_MPI
  for (int i = 0; i < bottom.size(); ++i) {
    vector<int> shape = bottom[i]->shape();
    shape[0] /= (Caffe::parallel_mode()==Caffe::MPI)?Caffe::MPI_all_rank():1;
    top[i]->Reshape(shape);

    if (Caffe::parallel_mode()!=Caffe::MPI){
      //if not in MPI mode, simply share data
      top[i]->ShareData(*bottom[i]);
      top[i]->ShareDiff(*bottom[i]);
    }
  }
#else
  for (int i = 0; i < bottom.size(); ++i){
    top[i]->ReshapeLike(*bottom[i]);
    top[i]->ShareData(*bottom[i]);
    top[i]->ShareDiff(*bottom[i]);
  }
#endif
}

template <typename Dtype>
void ScatterLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
#ifdef USE_MPI

  for (int i = 0; i < bottom.size(); ++i) {
    //Gather the bottom to the top
    caffe_iscatter((Dtype*)bottom[i]->cpu_data(),top[i]->mutable_cpu_data(), top[i]->count());
    mpi_force_synchronize();
  }
#else
#endif
}

template <typename Dtype>
void ScatterLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
#ifdef USE_MPI

  for (int i = 0; i < bottom.size(); ++i) {
    caffe_iallgather((Dtype*)top[i]->cpu_diff(),bottom[i]->mutable_cpu_diff(), top[i]->count());
    mpi_force_synchronize();
    //compensate the scale on diff IMPORTANT
    caffe_scal(bottom[i]->count(), Dtype(1)/Dtype(Caffe::MPI_all_rank()),
               bottom[i]->mutable_cpu_diff());
  }
#else
#endif
}

#ifdef CPU_ONLY
STUB_GPU(ScatterLayer);
#endif

INSTANTIATE_CLASS(ScatterLayer);
REGISTER_LAYER_CLASS(Scatter);

} // namespace caffe

