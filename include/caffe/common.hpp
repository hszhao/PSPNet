#ifndef CAFFE_COMMON_HPP_
#define CAFFE_COMMON_HPP_

#include <boost/shared_ptr.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <climits>
#include <cmath>
#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>  // pair
#include <vector>

#ifdef USE_MPI
  #include "mpi.h"

#define MPI_CHECK(cond) \
do { \
    int status = cond; \
    CHECK_EQ(status, MPI_SUCCESS) << " " \
      << "MPI Error Code: " << status; \
  } while (0)
#endif

#ifdef WITH_PYTHON_LAYER
#include <boost/python.hpp>
#endif

#include "caffe/util/device_alternate.hpp"

// gflags 2.1 issue: namespace google was changed to gflags without warning.
// Luckily we will be able to use GFLAGS_GFLAGS_H_ to detect if it is version
// 2.1. If yes, we will add a temporary solution to redirect the namespace.
// TODO(Yangqing): Once gflags solves the problem in a more elegant way, let's
// remove the following hack.
#ifndef GFLAGS_GFLAGS_H_
namespace gflags = google;
#endif  // GFLAGS_GFLAGS_H_

// Disable the copy and assignment operator for a class.
#define DISABLE_COPY_AND_ASSIGN(classname) \
private:\
  classname(const classname&);\
  classname& operator=(const classname&)

// Instantiate a class with float and double specifications.
#define INSTANTIATE_CLASS(classname) \
  char gInstantiationGuard##classname; \
  template class classname<float>; \
  template class classname<double>

#define INSTANTIATE_LAYER_GPU_FORWARD(classname) \
  template void classname<float>::Forward_gpu( \
      const std::vector<Blob<float>*>& bottom, \
      const std::vector<Blob<float>*>& top); \
  template void classname<double>::Forward_gpu( \
      const std::vector<Blob<double>*>& bottom, \
      const std::vector<Blob<double>*>& top);

#define INSTANTIATE_LAYER_GPU_BACKWARD(classname) \
  template void classname<float>::Backward_gpu( \
      const std::vector<Blob<float>*>& top, \
      const std::vector<bool>& propagate_down, \
      const std::vector<Blob<float>*>& bottom); \
  template void classname<double>::Backward_gpu( \
      const std::vector<Blob<double>*>& top, \
      const std::vector<bool>& propagate_down, \
      const std::vector<Blob<double>*>& bottom)

#define INSTANTIATE_LAYER_GPU_FUNCS(classname) \
  INSTANTIATE_LAYER_GPU_FORWARD(classname); \
  INSTANTIATE_LAYER_GPU_BACKWARD(classname)

// A simple macro to mark codes that are not implemented, so that when the code
// is executed we will see a fatal log.
#define NOT_IMPLEMENTED LOG(FATAL) << "Not Implemented Yet"

// See PR #1236
namespace cv { class Mat; }

namespace caffe {

// We will use the boost shared_ptr instead of the new C++11 one mainly
// because cuda does not work (at least now) well with C++11 features.
using boost::shared_ptr;

// Common functions and classes from std that caffe often uses.
using std::fstream;
using std::ios;
using std::isnan;
using std::isinf;
using std::iterator;
using std::make_pair;
using std::map;
using std::ostringstream;
using std::pair;
using std::set;
using std::string;
using std::stringstream;
using std::vector;

// A global initialization function that you should call in your main function.
// Currently it initializes google flags and google logging.
void GlobalInit(int* pargc, char*** pargv);

// A global function to clear up remaining stuffs
void GlobalFinalize();

// Header for system entropy source
int64_t cluster_seedgen(bool sync=true);

  // A singleton class to hold common caffe stuff, such as the handler that
// caffe is going to use for cublas, curand, etc.
class Caffe {
 public:
  ~Caffe();
  inline static Caffe& Get() {
    if (!singleton_.get()) {
      singleton_.reset(new Caffe());
    }
    return *singleton_;
  }
  enum Brew { CPU, GPU };

  // This random number generator facade hides boost and CUDA rng
  // implementation from one another (for cross-platform compatibility).
  class RNG {
   public:
    RNG();
    explicit RNG(unsigned int seed);
    explicit RNG(const RNG&);
    RNG& operator=(const RNG&);
    void* generator();
   private:
    class Generator;
    shared_ptr<Generator> generator_;
  };

  // Getters for boost rng, curand, and cublas handles
  inline static RNG& rng_stream() {
    if (!Get().random_generator_) {
      Get().random_generator_.reset(new RNG());
    }
    return *(Get().random_generator_);
  }
#ifndef CPU_ONLY
  inline static cublasHandle_t cublas_handle() { return Get().cublas_handle_; }
  inline static curandGenerator_t curand_generator() {
    return Get().curand_generator_;
  }
#endif

  // Returns the mode: running on CPU or GPU.
  inline static Brew mode() { return Get().mode_; }
  // The setters for the variables
  // Sets the mode. It is recommended that you don't change the mode halfway
  // into the program since that may cause allocation of pinned memory being
  // freed in a non-pinned way, which may cause problems - I haven't verified
  // it personally but better to note it here in the header file.
  inline static void set_mode(Brew mode) { Get().mode_ = mode; }
  // Sets the random seed of both boost and curand
  static void set_random_seed(const unsigned int seed);
  // Sets the device. Since we have cublas and curand stuff, set device also
  // requires us to reset those values.
  static void SetDevice(const int device_id);
  // Prints the current GPU status.
  static void DeviceQuery();
  inline static void setThreadId(int thread_id){Get().thread_id_ = thread_id;}
  inline static int getThreadId(){return Get().thread_id_;}
  inline static void setThreadNum(int thread_num){Get().thread_num_ = thread_num;}
  inline static int getThreadNum(){return Get().thread_num_;}
  inline static void setGPUId(int gpu_id){Get().gpu_id_ = gpu_id;}
  inline static int getGPUId(){return Get().gpu_id_;}
  inline static void setIterSize(int iter_size){Get().iter_size_ = iter_size;}
  inline static int getIterSize(){return Get().iter_size_;}
  inline static void setIter(int iter){Get().iter_ = iter;}
  inline static int getIter(){return Get().iter_;}
  inline static void setNodeNum(int node_num){Get().node_num_ = node_num;}
  inline static int getNodeNum(){return Get().node_num_;}
  inline static void setBestAccuracy(float best_accuracy){Get().best_accuracy_ = best_accuracy;}
  inline static float getBestAccuracy(){return Get().best_accuracy_;}
  inline static void setAccuracy(float accuracy){Get().accuracy_ = accuracy;}
  inline static float getAccuracy(){return Get().accuracy_;}
  inline static void setTaskList(void *task_list) {Get().task_list_ = task_list;}
  inline static void* getTaskList(){return Get().task_list_;}
  //inline static Strategy getStrategy() { return Get().strategy_; }
  //inline static void setStrategy(Strategy strategy) { Get().strategy_ = strategy; }
  inline static bool getMemoryOpt() { return Get().memory_opt_; }
  inline static void setMemoryOpt(const bool opt) { Get().memory_opt_ = opt; }

#ifdef USE_MPI
  enum PARALLEL_MODE { NO, MPI };

  //Returns current parallel mode, No or MPI
  inline static PARALLEL_MODE parallel_mode() {return Get().parallel_mode_;}
  // Setter of parallel mode
  inline static void set_parallel_mode(PARALLEL_MODE mode) {Get().parallel_mode_ = mode;}

  //Returns MPI_MY_RANK
  inline static int MPI_my_rank(){return Get().mpi_my_rank_;}
  inline static int MPI_all_rank(){return Get().mpi_all_rank_;}
  inline static void MPI_build_rank(){
    MPI_Comm_rank(MPI_COMM_WORLD, &(Get().mpi_my_rank_));
    MPI_Comm_size(MPI_COMM_WORLD, &(Get().mpi_all_rank_));
  }
  inline static int device_id(){return Get().device_id_;}
  inline static int remaining_sub_iter(){return Get().remaining_sub_iter_;}
  inline static void set_remaining_sub_iter(int n){Get().remaining_sub_iter_ = n;}
#endif

#ifdef WITH_PYTHON_LAYER
  inline static PyThreadState* py_tstate(){return Get().py_tstate_;}
  inline static void set_py_tstate(PyThreadState* new_state){Get().py_tstate_ = new_state;}
#endif

#ifdef USE_CUDNN
  inline static int cudnn_mem_richness(){return Get().cudnn_mem_richness_;}
  inline static void set_cudnn_mem_richness(int richness){Get().cudnn_mem_richness_ = richness;}
#endif

 protected:
#ifndef CPU_ONLY
  cublasHandle_t cublas_handle_;
  curandGenerator_t curand_generator_;
#endif
  shared_ptr<RNG> random_generator_;
  int thread_id_;
  int thread_num_;
  int node_num_;
  int gpu_id_;
  int iter_size_;
  int iter_;
  float best_accuracy_;
  float accuracy_;
  void* task_list_;
  //Strategy strategy_;
  bool memory_opt_;

#ifdef USE_CUDNN
  int cudnn_mem_richness_;
#endif

#ifdef USE_MPI

  PARALLEL_MODE parallel_mode_;
  int mpi_my_rank_;
  int mpi_all_rank_;
  int device_id_;
  int remaining_sub_iter_;
#endif

#ifdef WITH_PYTHON_LAYER
  PyThreadState* py_tstate_;
#endif

  Brew mode_;
  static shared_ptr<Caffe> singleton_;

 private:
  // The private constructor to avoid duplicate instantiation.
  Caffe();

  DISABLE_COPY_AND_ASSIGN(Caffe);
};

}  // namespace caffe

#endif  // CAFFE_COMMON_HPP_
