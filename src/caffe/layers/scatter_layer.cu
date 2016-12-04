#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/mpi_functions.hpp"

namespace caffe {

template <typename Dtype>
void ScatterLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
const vector<Blob<Dtype>*>& top) {

  #ifdef USE_MPI
  if (Caffe::parallel_mode() == Caffe::MPI){
    CUDA_CHECK(cudaDeviceSynchronize());
    for (int i = 0; i < bottom.size(); ++i) {
      //Gather the bottom to the top
      caffe_iscatter((Dtype*)bottom[i]->gpu_data(),top[i]->mutable_gpu_data(), top[i]->count());
      mpi_force_synchronize();
    }
  }
  #endif
  //Do nothing if not in MPI mode
}

template <typename Dtype>
void ScatterLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  #ifdef USE_MPI
    if (Caffe::parallel_mode() == Caffe::MPI){
      CUDA_CHECK(cudaDeviceSynchronize());
      for (int i = 0; i < bottom.size(); ++i) {
          //Scatter the top diff to buttom
          if (propagate_down[i]) {
          caffe_iallgather((Dtype*)top[i]->gpu_diff(),bottom[i]->mutable_gpu_diff(), top[i]->count());
          mpi_force_synchronize();
          //compensate the scale on diff IMPORTANT
          caffe_gpu_scal(bottom[i]->count(), Dtype(1)/Dtype(Caffe::MPI_all_rank()),
                         bottom[i]->mutable_gpu_diff());
        }
      }
    }
  #endif
}

INSTANTIATE_LAYER_GPU_FUNCS(ScatterLayer);

}  // namespace caffe
