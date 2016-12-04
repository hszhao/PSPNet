//
// Created by alex on 8/25/15.
//

#ifdef USE_MPI

#include "caffe/caffe.hpp"
#include "caffe/util/mpi_functions.hpp"
#include "caffe/util/channel.hpp"

namespace caffe {
  template <typename Dtype>
  void caffe_iallreduce(Dtype* data, int count){
    MPIJob job = {data, data, count, sizeof(Dtype), OP_SUM_ALL};
    MPIComm::AddMPIJob(job);
  }

  template void caffe_iallreduce<float>(float* data, int count);
  template void caffe_iallreduce<double>(double* data, int count);

  template <typename Dtype>
  void caffe_iallreduce(Dtype* src_data, Dtype* dst_data, int count){
    MPIJob job = {src_data, dst_data, count, sizeof(Dtype), OP_SUM_ALL};
    MPIComm::AddMPIJob(job);
  }

  template void caffe_iallreduce<float>(float* src_data, float* dst_data, int count);
  template void caffe_iallreduce<double>(double* src_data, double* dst_data, int count);

  template <typename Dtype>
  void caffe_iallgather(Dtype* src_data, Dtype* dst_data, int count){
    MPIJob job = {src_data, dst_data, count, sizeof(Dtype), OP_GATHER};
    MPIComm::AddMPIJob(job);
  }
  template void caffe_iallgather<float>(float*, float*, int);
  template void caffe_iallgather<double>(double*, double*, int);

  template <typename Dtype>
  void caffe_iscatter(Dtype* src_data, Dtype* dst_data, int count){
    MPIJob job = {src_data, dst_data, count, sizeof(Dtype), OP_SCATTER};
    MPIComm::AddMPIJob(job);
  }

  template void caffe_iscatter<float>(float*, float*, int);
  template void caffe_iscatter<double>(double*, double*, int);

  template <typename Dtype>
  void caffe_ibcast(Dtype* data, int count){
    MPIJob job = {data, data, count, sizeof(Dtype), OP_BROADCAST};
    MPIComm::AddMPIJob(job);
  }
  template void caffe_ibcast<float>(float* data, int count);
  template void caffe_ibcast<double>(double* data, int count);

  void mpi_force_synchronize(){
    MPIComm::Syncrhonize();
  }
}

#endif //USE_MPI
