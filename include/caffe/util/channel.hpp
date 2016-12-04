//
// Created by alex on 8/25/15.
//

#ifndef CAFFE_CHANNEL_HPP
#define CAFFE_CHANNEL_HPP

#ifdef USE_MPI

#include <boost/shared_ptr.hpp>
#include <boost/atomic.hpp>
#include <boost/thread.hpp>
#include <queue>

using std::queue;
using boost::mutex;
using boost::condition_variable;
using boost::shared_ptr;
using boost::atomic;

namespace caffe {

enum OperationType {
    OP_SUM_ALL, OP_GATHER, OP_SCATTER, OP_BROADCAST
};

class MPIJob {
public:
  void* src_ptr_; // str_ptr_==NULL indicates IN_PLACE operation
  void* dst_ptr_;
  int count_;
  int dtype_size_;
  OperationType op_;
};

class MPIComm{
  public:
    ~MPIComm();
    inline static MPIComm& Get() {
      if (!singleton_.get()) {
        singleton_.reset(new MPIComm());
        singleton_->StartProcessing();
      }
      return *singleton_;
    }

    inline static void AddMPIJob(MPIJob job){ Get().AddJob(job);};
    inline static void Syncrhonize(){Get().WaitAll();}

  private:
    MPIComm();

    void ThreadFunc();
    void DispatchJob(MPIJob& job);
    bool IsRunning();
    bool IsIdle();
    void StartProcessing();
    void EndProcessing();
    void AddJob(MPIJob new_job);
    void WaitAll();

    queue<MPIJob> task_queue_;
    mutable mutex queue_mutex_;
    atomic<bool> running_, started_;
    shared_ptr<boost::thread> thread_;
    condition_variable cond_work_;
    condition_variable cond_finish_;

    static shared_ptr<MPIComm> singleton_;

};
};

#endif //USE_MPI

#endif //CAFFE_CHANNEL_HPP
