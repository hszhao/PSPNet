//
// Created by alex on 8/25/15.
//

#ifndef CAFFE_MPI_FUNCTIONS_HPP
#define CAFFE_MPI_FUNCTIONS_HPP

namespace caffe {
  template <typename Dtype>
  void caffe_iallreduce(Dtype* data, int count);

  template <typename Dtype>
  void caffe_iallreduce(Dtype* src_data, Dtype* dst_data, int count);

  template <typename Dtype>
  void caffe_iallgather(Dtype* src_data, Dtype* dst_data, int count);

  template <typename Dtype>
  void caffe_iscatter(Dtype* src_data, Dtype* dst_data, int count);

  template <typename Dtype>
  void caffe_ibcast(Dtype* data, int count);

  void mpi_force_synchronize();


}

#endif //CAFFE_MPI_FUNCTIONS_HPP_HPP
