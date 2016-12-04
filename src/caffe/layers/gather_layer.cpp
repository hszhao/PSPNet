#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/mpi_functions.hpp"

namespace caffe {

template <typename Dtype>
void GatherLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                    const vector<Blob<Dtype>*>& top) {
  //Sanity check
  CHECK_EQ(bottom.size(), bottom.size())<<"Must have equal number of top and bottom blobs";


}

  template <typename Dtype>
void GatherLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                  const vector<Blob<Dtype>*>& top) {
#ifdef USE_MPI
    for (int i = 0; i < bottom.size(); ++i){
      vector<int> gathered_shape(bottom[i]->shape());
      gathered_shape[0] *= (Caffe::parallel_mode()==Caffe::MPI)?Caffe::MPI_all_rank():1;
      top[i]->Reshape(gathered_shape);

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
void GatherLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                    const vector<Blob<Dtype>*>& top) {
  #ifdef USE_MPI
  if (Caffe::parallel_mode() == Caffe::MPI){
    for (int i = 0; i < bottom.size(); ++i) {
      //Gather the bottom to the top
      caffe_iallgather((Dtype*)bottom[i]->cpu_data(),top[i]->mutable_cpu_data(), bottom[i]->count());
      mpi_force_synchronize();
    }
  }
  #endif
  //Do nothing if not if MPI mode
}

template <typename Dtype>
void GatherLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                     const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  #ifdef USE_MPI
    if (Caffe::parallel_mode() == Caffe::MPI){
      for (int i = 0; i < bottom.size(); ++i) {
        //Gather the bottom to the top
        if (propagate_down[i]) {
          caffe_iscatter((Dtype*)top[i]->cpu_diff(),bottom[i]->mutable_cpu_diff(), bottom[i]->count());
          mpi_force_synchronize();
          //compensate the scale on diff IMPORTANT
          caffe_scal(bottom[i]->count(), Dtype(Caffe::MPI_all_rank()),
                         bottom[i]->mutable_cpu_diff());
        }
      }
    }
  #endif
}

#ifdef CPU_ONLY
STUB_GPU(GatherLayer);
#endif

INSTANTIATE_CLASS(GatherLayer);
REGISTER_LAYER_CLASS(Gather);
}
