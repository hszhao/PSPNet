#include <glog/logging.h>
#include <cstdio>
#include <ctime>

#include "caffe/common.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

shared_ptr<Caffe> Caffe::singleton_;

// random seeding
int64_t cluster_seedgen(bool sync) {
  int64_t s, seed, pid;
  FILE* f = fopen("/dev/urandom", "rb");
  if (f && fread(&seed, 1, sizeof(seed), f) == sizeof(seed)) {
    fclose(f);
    return seed;
  }

  LOG(INFO) << "System entropy source not available, "
              "using fallback algorithm to generate seed instead.";
  if (f)
    fclose(f);

  pid = getpid();
  s = time(NULL);
  seed = abs(((s * 181) * ((pid - 83) * 359)) % 104729);
  return seed;
}


void GlobalInit(int* pargc, char*** pargv) {

  // Google flags.
  ::gflags::ParseCommandLineFlags(pargc, pargv, true);
  // Google logging.
  ::google::InitGoogleLogging(*(pargv)[0]);
  // Provide a backtrace on segfault.
  ::google::InstallFailureSignalHandler();

#ifdef USE_MPI
  //try start MPI communication system
  int provided_thread_support;
  MPI_Init_thread(pargc, pargv, MPI_THREAD_MULTIPLE, &provided_thread_support);

  CHECK_GE(provided_thread_support, MPI_THREAD_SERIALIZED)<<" Cannot activate MPI thread support";

  Caffe::MPI_build_rank();

  if (Caffe::MPI_all_rank() > 1) {
    Caffe::set_parallel_mode(Caffe::MPI);
    LOG(INFO)<<"Running parallel training with MPI support!";
  }else{
    Caffe::set_parallel_mode(Caffe::NO);
    LOG(INFO)<<"You are running caffe compiled with MPI support. Now it's running in non-parallel model";
  }

  //disable slave processes from logging to stderr
  //also enable logging only events above ERROR level to logfile.
  if (Caffe::MPI_my_rank() != 0){
    FLAGS_logtostderr = false;
    FLAGS_minloglevel = 2;
  }
  
  int thread_id=0,thread_num=0,node_num=0,gpus_per_node=0;
  int color = 0, key = 0, count = 0, name_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &thread_id);
  MPI_Comm_size(MPI_COMM_WORLD, &thread_num);

  char node_name[MPI::MAX_PROCESSOR_NAME];
  MPI::Get_processor_name(node_name, name_size);
  char *all_node_name = (char *)malloc(sizeof(char) * MPI::MAX_PROCESSOR_NAME * thread_num);
  MPI_Allgather(node_name,MPI::MAX_PROCESSOR_NAME,MPI_CHAR,all_node_name,
      MPI::MAX_PROCESSOR_NAME,MPI_CHAR,MPI_COMM_WORLD); 
  MPI_Comm dup_comm, sub_comm;
  for(int i = 0; i < thread_num; i++){
    if (i == 0) node_num = 1;
    else if(strcmp(all_node_name + MPI::MAX_PROCESSOR_NAME * i,all_node_name + MPI::MAX_PROCESSOR_NAME * (i - 1)) != 0){
      node_num++;
    }
  }
  for(int i = 0; i < thread_num; i++){
    if(strcmp(node_name,all_node_name + MPI::MAX_PROCESSOR_NAME * i) == 0)
    {
       if(count == 0)
           color = i;
       if(i == thread_id)
       {
         key = count;
         break;
       }
       count++;
    }
  }

  MPI_Comm_dup(MPI_COMM_WORLD,&dup_comm);
  MPI_Comm_split(dup_comm,color,key,&sub_comm);

  int sub_thread_num = 0, _MIN_sub_thread_id = 0;
  MPI_Comm_size(sub_comm, &sub_thread_num);
  MPI_Allreduce(&thread_id, &_MIN_sub_thread_id, 1, MPI_INT, MPI_MIN, sub_comm);

  LOG(INFO)<<"_MIN sub thread id:"<<_MIN_sub_thread_id<<" node name:"<<node_name;

  CUDA_CHECK(cudaGetDeviceCount(&gpus_per_node));
  if(FLAGS_gpu >=0 && FLAGS_gpu <= gpus_per_node) {
    FLAGS_gpu += (thread_id - _MIN_sub_thread_id);
    LOG(INFO)<<"FLAGS_gpu:"<<FLAGS_gpu;
    Caffe::SetDevice(FLAGS_gpu);
    Caffe::set_mode(Caffe::GPU);
  }

  Caffe::setGPUId(FLAGS_gpu);

  Caffe::setThreadId(thread_id);
  Caffe::setThreadNum(thread_num); 
  Caffe::setNodeNum(node_num);
#endif

}

void GlobalFinalize(){
  //Add something here

  #ifdef USE_MPI
  MPI_Finalize();
  #endif
}

#ifdef CPU_ONLY  // CPU-only Caffe.

Caffe::Caffe()
    : random_generator_(), mode_(Caffe::CPU) { }

Caffe::~Caffe() { }

void Caffe::set_random_seed(const unsigned int seed) {
  // RNG seed
  Get().random_generator_.reset(new RNG(seed));
}

void Caffe::SetDevice(const int device_id) {
  NO_GPU;
}

void Caffe::DeviceQuery() {
  NO_GPU;
}


class Caffe::RNG::Generator {
 public:
  Generator() : rng_(new caffe::rng_t(cluster_seedgen())) {}
  explicit Generator(unsigned int seed) : rng_(new caffe::rng_t(seed)) {}
  caffe::rng_t* rng() { return rng_.get(); }
 private:
  shared_ptr<caffe::rng_t> rng_;
};

Caffe::RNG::RNG() : generator_(new Generator()) { }

Caffe::RNG::RNG(unsigned int seed) : generator_(new Generator(seed)) { }

Caffe::RNG& Caffe::RNG::operator=(const RNG& other) {
  generator_ = other.generator_;
  return *this;
}

void* Caffe::RNG::generator() {
  return static_cast<void*>(generator_->rng());
}

#else  // Normal GPU + CPU Caffe.

Caffe::Caffe()
    : cublas_handle_(NULL), curand_generator_(NULL), random_generator_(),
    mode_(Caffe::CPU) {
  #ifndef USE_MPI
  // Try to create a cublas handler, and report an error if failed (but we will
  // keep the program running as one might just want to run CPU code).
  if (cublasCreate(&cublas_handle_) != CUBLAS_STATUS_SUCCESS) {
    LOG(ERROR) << "Cannot create Cublas handle. Cublas won't be available.";
  }
  // Try to create a curand handler.
  if (curandCreateGenerator(&curand_generator_, CURAND_RNG_PSEUDO_DEFAULT)
      != CURAND_STATUS_SUCCESS ||
      curandSetPseudoRandomGeneratorSeed(curand_generator_, cluster_seedgen())
      != CURAND_STATUS_SUCCESS) {
    LOG(ERROR) << "Cannot create Curand generator. Curand won't be available.";
  }
  #else
  // we are not trying to create the any cuda stuff here
  // because on exclusive mode GPUs it will cause program fail
  // Reason: no device id assigned at this time, all processes will try to access gpu 0.
  #endif

  #ifdef USE_CUDNN
    cudnn_mem_richness_ = 1;
  #endif

  #ifdef WITH_PYTHON_LAYER
  py_tstate_ = NULL;
  #endif
}

Caffe::~Caffe() {
  if (cublas_handle_) CUBLAS_CHECK(cublasDestroy(cublas_handle_));
  if (curand_generator_) {
    CURAND_CHECK(curandDestroyGenerator(curand_generator_));
  }
}

void Caffe::set_random_seed(const unsigned int seed) {
  // Curand seed
  static bool g_curand_availability_logged = false;
  if (Get().curand_generator_) {
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(curand_generator(),
        seed));
    CURAND_CHECK(curandSetGeneratorOffset(curand_generator(), 0));
  } else {
    if (!g_curand_availability_logged) {
        LOG(ERROR) <<
            "Curand not available. Skipping setting the curand seed.";
        g_curand_availability_logged = true;
    }
  }
  // RNG seed
  Get().random_generator_.reset(new RNG(seed));
}

void Caffe::SetDevice(const int device_id) {
  int current_device;
  std::cout<<"Setting device "<<device_id<<"\n";
  CUDA_CHECK(cudaGetDevice(&current_device));
  if (current_device == device_id && Get().cublas_handle_ && Get().curand_generator_) {
    return;
  }
  // The call to cudaSetDevice must come before any calls to Get, which
  // may perform initialization using the GPU.
  CUDA_CHECK(cudaSetDevice(device_id));
  if (Get().cublas_handle_) CUBLAS_CHECK(cublasDestroy(Get().cublas_handle_));
  if (Get().curand_generator_) {
    CURAND_CHECK(curandDestroyGenerator(Get().curand_generator_));
  }
  CUBLAS_CHECK(cublasCreate(&Get().cublas_handle_));
  CURAND_CHECK(curandCreateGenerator(&Get().curand_generator_,
      CURAND_RNG_PSEUDO_DEFAULT));
  CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(Get().curand_generator_,
      cluster_seedgen()));
#ifdef USE_MPI
  Get().device_id_ = device_id;
#endif
}

void Caffe::DeviceQuery() {
  cudaDeviceProp prop;
  int device;
  if (cudaSuccess != cudaGetDevice(&device)) {
    printf("No cuda device present.\n");
    return;
  }
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  LOG(INFO) << "Device id:                     " << device;
  LOG(INFO) << "Major revision number:         " << prop.major;
  LOG(INFO) << "Minor revision number:         " << prop.minor;
  LOG(INFO) << "Name:                          " << prop.name;
  LOG(INFO) << "Total global memory:           " << prop.totalGlobalMem;
  LOG(INFO) << "Total shared memory per block: " << prop.sharedMemPerBlock;
  LOG(INFO) << "Total registers per block:     " << prop.regsPerBlock;
  LOG(INFO) << "Warp size:                     " << prop.warpSize;
  LOG(INFO) << "Maximum memory pitch:          " << prop.memPitch;
  LOG(INFO) << "Maximum threads per block:     " << prop.maxThreadsPerBlock;
  LOG(INFO) << "Maximum dimension of block:    "
      << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", "
      << prop.maxThreadsDim[2];
  LOG(INFO) << "Maximum dimension of grid:     "
      << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", "
      << prop.maxGridSize[2];
  LOG(INFO) << "Clock rate:                    " << prop.clockRate;
  LOG(INFO) << "Total constant memory:         " << prop.totalConstMem;
  LOG(INFO) << "Texture alignment:             " << prop.textureAlignment;
  LOG(INFO) << "Concurrent copy and execution: "
      << (prop.deviceOverlap ? "Yes" : "No");
  LOG(INFO) << "Number of multiprocessors:     " << prop.multiProcessorCount;
  LOG(INFO) << "Kernel execution timeout:      "
      << (prop.kernelExecTimeoutEnabled ? "Yes" : "No");
  return;
}


class Caffe::RNG::Generator {
 public:
  Generator() : rng_(new caffe::rng_t(cluster_seedgen())) {}
  explicit Generator(unsigned int seed) : rng_(new caffe::rng_t(seed)) {}
  caffe::rng_t* rng() { return rng_.get(); }
 private:
  shared_ptr<caffe::rng_t> rng_;
};

Caffe::RNG::RNG() : generator_(new Generator()) { }

Caffe::RNG::RNG(unsigned int seed) : generator_(new Generator(seed)) { }

Caffe::RNG& Caffe::RNG::operator=(const RNG& other) {
  generator_.reset(other.generator_.get());
  return *this;
}

void* Caffe::RNG::generator() {
  return static_cast<void*>(generator_->rng());
}

const char* cublasGetErrorString(cublasStatus_t error) {
  switch (error) {
  case CUBLAS_STATUS_SUCCESS:
    return "CUBLAS_STATUS_SUCCESS";
  case CUBLAS_STATUS_NOT_INITIALIZED:
    return "CUBLAS_STATUS_NOT_INITIALIZED";
  case CUBLAS_STATUS_ALLOC_FAILED:
    return "CUBLAS_STATUS_ALLOC_FAILED";
  case CUBLAS_STATUS_INVALID_VALUE:
    return "CUBLAS_STATUS_INVALID_VALUE";
  case CUBLAS_STATUS_ARCH_MISMATCH:
    return "CUBLAS_STATUS_ARCH_MISMATCH";
  case CUBLAS_STATUS_MAPPING_ERROR:
    return "CUBLAS_STATUS_MAPPING_ERROR";
  case CUBLAS_STATUS_EXECUTION_FAILED:
    return "CUBLAS_STATUS_EXECUTION_FAILED";
  case CUBLAS_STATUS_INTERNAL_ERROR:
    return "CUBLAS_STATUS_INTERNAL_ERROR";
#if CUDA_VERSION >= 6000
  case CUBLAS_STATUS_NOT_SUPPORTED:
    return "CUBLAS_STATUS_NOT_SUPPORTED";
#endif
#if CUDA_VERSION >= 6050
  case CUBLAS_STATUS_LICENSE_ERROR:
    return "CUBLAS_STATUS_LICENSE_ERROR";
#endif
  }
  return "Unknown cublas status";
}

const char* curandGetErrorString(curandStatus_t error) {
  switch (error) {
  case CURAND_STATUS_SUCCESS:
    return "CURAND_STATUS_SUCCESS";
  case CURAND_STATUS_VERSION_MISMATCH:
    return "CURAND_STATUS_VERSION_MISMATCH";
  case CURAND_STATUS_NOT_INITIALIZED:
    return "CURAND_STATUS_NOT_INITIALIZED";
  case CURAND_STATUS_ALLOCATION_FAILED:
    return "CURAND_STATUS_ALLOCATION_FAILED";
  case CURAND_STATUS_TYPE_ERROR:
    return "CURAND_STATUS_TYPE_ERROR";
  case CURAND_STATUS_OUT_OF_RANGE:
    return "CURAND_STATUS_OUT_OF_RANGE";
  case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
    return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
  case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
    return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
  case CURAND_STATUS_LAUNCH_FAILURE:
    return "CURAND_STATUS_LAUNCH_FAILURE";
  case CURAND_STATUS_PREEXISTING_FAILURE:
    return "CURAND_STATUS_PREEXISTING_FAILURE";
  case CURAND_STATUS_INITIALIZATION_FAILED:
    return "CURAND_STATUS_INITIALIZATION_FAILED";
  case CURAND_STATUS_ARCH_MISMATCH:
    return "CURAND_STATUS_ARCH_MISMATCH";
  case CURAND_STATUS_INTERNAL_ERROR:
    return "CURAND_STATUS_INTERNAL_ERROR";
  }
  return "Unknown curand status";
}

#endif  // CPU_ONLY

}  // namespace caffe
