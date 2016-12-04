#ifdef USE_CUDNN
#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

#include <boost/unordered_map.hpp>
#include <cudnn.h>

using boost::unordered_map;

namespace caffe {

// Set to three for the benefit of the backward pass, which
// can use separate streams for calculating the gradient w.r.t.
// bias, filter weights, and bottom data for each group independently
#define CUDNN_FWD_STREAMS_PER_GROUP 1
#define CUDNN_BWD_STREAMS_PER_GROUP 2

template <typename Dtype>
unordered_map<CuDNNConvolutionLayer<Dtype>*, typename CuDNNConvolutionLayer<Dtype>::PerfReg*> CuDNNConvolutionLayer<Dtype>::perf_reg;

template <typename Dtype>
bool CuDNNConvolutionLayer<Dtype>::need_optimize_ = true;


typedef struct {
    float total_time;
    vector<int> choices;
}MemRecord;

void updateDict(unordered_map<size_t, MemRecord>& dict, const size_t key, const float time, const vector<int>& choices){
  MemRecord rec;
  rec.total_time = time;
  rec.choices = choices;
  dict[key] = rec;
}

template <typename Dtype, typename PerfType>
void runTransitFunc(unordered_map<size_t, MemRecord>& new_dict, unordered_map<size_t, MemRecord>& prev_dict,
              const vector<PerfType>& perf, const size_t mem_limit){
  new_dict.clear();
  int mem_tick = (Caffe::cudnn_mem_richness()>0)?Caffe::cudnn_mem_richness() * 1000 : 1000;
  for (size_t i_algo = 0; i_algo < perf.size(); ++i_algo){
    PerfType algo_perf = perf[i_algo];
    size_t mem = (algo_perf.memory + mem_tick -1) / mem_tick ;
    float time = algo_perf.time;
    if (time < 0){
      continue;
    }
    for (unordered_map<size_t, MemRecord>::iterator mc = prev_dict.begin(); mc != prev_dict.end(); ++mc){
      size_t new_mem = mc->first + mem;

      if (new_mem > mem_limit){
        continue;
      }
      float new_time = mc->second.total_time + time;
      bool update = false;
      if (new_dict.find(new_mem) == new_dict.end()){
        update = true;
      }else{
        MemRecord& ext_rec = new_dict[new_mem];
        if (ext_rec.total_time > new_time){
          update = true;
        }
      }

      if (update) {
        vector<int> ch = mc->second.choices;
        ch.push_back(i_algo);
        updateDict(new_dict, new_mem, new_time, ch);
        //LOG(INFO)<<new_mem;
      }
    }
  }
  prev_dict = new_dict;
};

template <typename Dtype, typename PerfType>
void initTransitFunc(unordered_map<size_t, MemRecord>& new_dict,
                    const vector<PerfType>& perf, const size_t mem_limit){
  new_dict.clear();
  int mem_tick = (Caffe::cudnn_mem_richness()>0)?Caffe::cudnn_mem_richness() * 1000 : 1000;
  for (size_t i_algo = 0; i_algo < perf.size(); ++i_algo){
    PerfType algo_perf = perf[i_algo];
    size_t mem = (algo_perf.memory + mem_tick -1) / mem_tick;
    if (mem > mem_limit) continue;
    float time = algo_perf.time;
    if (time < 0){
      continue;
    }
    if (new_dict.find(mem) == new_dict.end()){
      vector<int> tmp;
      tmp.push_back(i_algo);
      updateDict(new_dict, mem, time, tmp);
    }else{
      //check and update
      MemRecord& rec = new_dict[mem];
      if (time < rec.total_time){
        //update
        vector<int> tmp;
        tmp.push_back(i_algo);
        updateDict(new_dict,mem, time, tmp);
      }
    }
  }
};

template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::RuntimeOptimize(size_t mem_limit) {

  if (!need_optimize_){
    return;
  }
  unordered_map<size_t, MemRecord> prev_dict;
  unordered_map<size_t, MemRecord> new_dict;

  mem_limit *= (Caffe::cudnn_mem_richness() > 0);
  //iterate
  for (typename unordered_map<CuDNNConvolutionLayer *, PerfReg *>::iterator layer_reg = perf_reg.begin();
       layer_reg != perf_reg.end(); ++layer_reg) {
    PerfReg &layer_perf = *(layer_reg->second);

    //foward
    for (int x = 0; x < layer_perf.fwd_perf.size(); ++x)
      if (prev_dict.size() == 0) {
        initTransitFunc<Dtype, cudnnConvolutionFwdAlgoPerf_t>(prev_dict, layer_perf.fwd_perf[x], mem_limit);
      } else
        runTransitFunc<Dtype, cudnnConvolutionFwdAlgoPerf_t>(new_dict, prev_dict, layer_perf.fwd_perf[x], mem_limit);

    //bwd filter
    for (int x = 0; x < layer_perf.bwd_filter_perf.size(); ++x) {
      runTransitFunc<Dtype, cudnnConvolutionBwdFilterAlgoPerf_t>(new_dict,
                                                                 prev_dict,
                                                                 layer_perf.bwd_filter_perf[x],
                                                                 mem_limit);
    }
    //bwd data
    for (int x = 0; x < layer_perf.bwd_data_perf.size(); ++x)
      runTransitFunc<Dtype, cudnnConvolutionBwdDataAlgoPerf_t>(new_dict,
                                                               prev_dict,
                                                               layer_perf.bwd_data_perf[x],
                                                               mem_limit);
  }

  // find optimal
  MemRecord *min_rec = &prev_dict.begin()->second;
  for (unordered_map<size_t, MemRecord>::iterator mc = prev_dict.begin(); mc != prev_dict.end(); ++mc) {
    if (mc->second.total_time < min_rec->total_time) {
      min_rec = &mc->second;
    }
  }

  //set optimal result
  vector<int> &choices = min_rec->choices;
  int cnt = 0;
  for (typename unordered_map<CuDNNConvolutionLayer *, PerfReg *>::iterator layer_reg = perf_reg.begin();
       layer_reg != perf_reg.end(); ++layer_reg){
    PerfReg &layer_perf = *(layer_reg->second);
    for (int x = 0; x < layer_perf.fwd_perf.size(); ++x) {
      layer_perf.fwd_algo[x] = choices[cnt++];
    }

    for (int x = 0; x < layer_perf.bwd_filter_perf.size(); ++x) {
      layer_perf.bwd_filter_algo[x] = choices[cnt++];
    }

    for (int x = 0; x < layer_perf.fwd_perf.size(); ++x) {
      layer_perf.bwd_data_algo[x] = choices[cnt++];
    }
    layer_reg->first->AdjustWorkSpaces();
  }

  need_optimize_ = false;
  LOG(INFO)<<"Optimized cudnn conv";
}

/**
 * TODO(dox) explain cuDNN interface
 */
template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  ConvolutionLayer<Dtype>::LayerSetUp(bottom, top);
  // Initialize CUDA streams and cuDNN.
  int total_streams_per_group = CUDNN_FWD_STREAMS_PER_GROUP + CUDNN_BWD_STREAMS_PER_GROUP;
  stream_         = new cudaStream_t[this->group_ * total_streams_per_group];
  handle_         = new cudnnHandle_t[this->group_ * total_streams_per_group];

  // Initialize algorithm arrays
  fwd_algo_       = new cudnnConvolutionFwdAlgo_t[bottom.size()];
  bwd_filter_algo_= new cudnnConvolutionBwdFilterAlgo_t[bottom.size()];
  bwd_data_algo_  = new cudnnConvolutionBwdDataAlgo_t[bottom.size()];

  // initialize size arrays
  workspace_fwd_sizes_ = new size_t[bottom.size()];
  workspace_bwd_filter_sizes_ = new size_t[bottom.size()];
  workspace_bwd_data_sizes_ = new size_t[bottom.size()];

  // initilized perf reg
  layer_perf_.bwd_filter_perf.resize(bottom.size());
  layer_perf_.bwd_data_perf.resize(bottom.size());
  layer_perf_.fwd_perf.resize(bottom.size());

  // register the layer to cudnn conv registry for global planning
  perf_reg[this] = &layer_perf_;


  layer_perf_.bwd_filter_algo.resize(bottom.size());
  layer_perf_.bwd_data_algo.resize(bottom.size());
  layer_perf_.fwd_algo.resize(bottom.size());


  // workspace data sizes start with zero
  workspaceSizeInBytes_fwd = workspaceSizeInBytes_bwd = 0;
  for (int i = 0; i < this->group_*CUDNN_FWD_STREAMS_PER_GROUP; ++i)
    workspaceData_fwd.push_back(shared_ptr<SyncedMemory>(new SyncedMemory()));
  for (int i = 0; i < this->group_*CUDNN_BWD_STREAMS_PER_GROUP; ++i)
    workspaceData_bwd_filter.push_back(shared_ptr<SyncedMemory>(new SyncedMemory()));
  for (int i = 0; i < this->group_*CUDNN_BWD_STREAMS_PER_GROUP; ++i)
    workspaceData_bwd_data.push_back(shared_ptr<SyncedMemory>(new SyncedMemory()));


  for (size_t i = 0; i < bottom.size(); ++i) {
    // initialize all to default algorithms
    fwd_algo_[i] = (cudnnConvolutionFwdAlgo_t)0;
    bwd_filter_algo_[i] = (cudnnConvolutionBwdFilterAlgo_t)0;
    bwd_data_algo_[i] = (cudnnConvolutionBwdDataAlgo_t)0;
    // default algorithms don't require workspace
    workspace_fwd_sizes_[i] = 0;
    workspace_bwd_data_sizes_[i] = 0;
    workspace_bwd_filter_sizes_[i] = 0;
  }

  for (int g = 0; g < this->group_ * total_streams_per_group; g++) {
    CUDA_CHECK(cudaStreamCreate(&stream_[g]));
    CUDNN_CHECK(cudnnCreate(&handle_[g]));
    CUDNN_CHECK(cudnnSetStream(handle_[g], stream_[g]));
  }

  // Set the indexing parameters.
  weight_offset_ = (this->num_output_ / this->group_)
      * (this->channels_ / this->group_) * this->kernel_h_ * this->kernel_w_;
  bias_offset_ = (this->num_output_ / this->group_);

  // Create filter descriptor.
  cudnn::createFilterDesc<Dtype>(&filter_desc_,
      this->num_output_ / this->group_, this->channels_ / this->group_,
      this->kernel_h_, this->kernel_w_);

  // Create tensor descriptor(s) for data and corresponding convolution(s).
  for (int i = 0; i < bottom.size(); i++) {
    cudnnTensorDescriptor_t bottom_desc;
    cudnn::createTensor4dDesc<Dtype>(&bottom_desc);
    bottom_descs_.push_back(bottom_desc);
    cudnnTensorDescriptor_t top_desc;
    cudnn::createTensor4dDesc<Dtype>(&top_desc);
    top_descs_.push_back(top_desc);
    cudnnConvolutionDescriptor_t conv_desc;
    cudnn::createConvolutionDesc<Dtype>(&conv_desc);
    conv_descs_.push_back(conv_desc);
  }

  // Tensor descriptor for bias.
  if (this->bias_term_) {
    cudnn::createTensor4dDesc<Dtype>(&bias_desc_);
  }

  handles_setup_ = true;

  prev_bottom_shapes_.resize(bottom.size(), vector<int>());
}

template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  ConvolutionLayer<Dtype>::Reshape(bottom, top);
  bottom_offset_ = (this->channels_ / this->group_)
      * this->height_ * this->width_;
  top_offset_ = (this->num_output_ / this->group_)
      * this->height_out_ * this->width_out_;

  // Specify workspace limit for kernels directly until we have a
  // planning strategy and a rewrite of Caffe's GPU memory mangagement.
  //
  // However this can be tuned by the "richness" parameter in the solver protobuf
  // By setting richness, you can increase the memory available to cuDNN and thus
  // let it choose fast but space consuming algorithms.
  for (int i = 0; i < bottom.size(); i++) {
    if (prev_bottom_shapes_[i] == bottom[i]->shape()) continue;
    prev_bottom_shapes_[i] = bottom[i]->shape();

    cudnn::setTensor4dDesc<Dtype>(&bottom_descs_[i],
                                  this->num_,
                                  this->channels_ / this->group_,
                                  this->height_, this->width_,
                                  this->channels_ * this->height_ * this->width_,
                                  this->height_ * this->width_,
                                  this->width_, 1);
    cudnn::setTensor4dDesc<Dtype>(&top_descs_[i],
                                  this->num_,
                                  this->num_output_ / this->group_,
                                  this->height_out_, this->width_out_,
                                  this->num_output_ * this->height_out_ * this->width_out_,
                                  this->height_out_ * this->width_out_,
                                  this->width_out_, 1);
    cudnn::setConvolutionDesc<Dtype>(&conv_descs_[i], bottom_descs_[i],
                                     filter_desc_, this->pad_h_, this->pad_w_,
                                     this->stride_h_, this->stride_w_);

    // choose forward and backward algorithms + workspace(s)
    const int kRequestedForwardAlgoCount = 6;
    vector<cudnnConvolutionFwdAlgoPerf_t> fwd_perf;
    fwd_perf.resize(kRequestedForwardAlgoCount);
    int returnedAlgoCount;
    CUDNN_CHECK(cudnnFindConvolutionForwardAlgorithm(handle_[0],
                                                     bottom_descs_[i],
                                                     filter_desc_,
                                                     conv_descs_[i],
                                                     top_descs_[i],
                                                     kRequestedForwardAlgoCount,
                                                     &returnedAlgoCount,
                                                     &fwd_perf[0]));
    layer_perf_.fwd_perf[i] =
        vector<cudnnConvolutionFwdAlgoPerf_t>(fwd_perf.begin(), fwd_perf.begin() + returnedAlgoCount);


    // choose backward algorithm for filter
    const int kRequestedBackwardFilterAlgoCount = 4;
    vector<cudnnConvolutionBwdFilterAlgoPerf_t> bwd_filter_perf;
    bwd_filter_perf.resize(kRequestedBackwardFilterAlgoCount);
    CUDNN_CHECK(cudnnFindConvolutionBackwardFilterAlgorithm(handle_[0],
                                                            bottom_descs_[i],
                                                            top_descs_[i],
                                                            conv_descs_[i],
                                                            filter_desc_,
                                                            kRequestedBackwardFilterAlgoCount,
                                                            &returnedAlgoCount,
                                                            &bwd_filter_perf[0]));
    layer_perf_.bwd_filter_perf[i] = vector<cudnnConvolutionBwdFilterAlgoPerf_t>(bwd_filter_perf.begin(),
                                                                                 bwd_filter_perf.begin()
                                                                                     + returnedAlgoCount);
    if (layer_perf_.bwd_filter_perf[i][0].algo == 2){
      LOG(INFO)<<"fft context time "<<layer_perf_.bwd_filter_perf[i][0].time<<" mem "<<layer_perf_.bwd_filter_perf[i][0].memory;
    }

    // choose backward algo for data
    const int kRequestedBackwardDataAlgoCount = 4;
    vector<cudnnConvolutionBwdDataAlgoPerf_t> bwd_data_perf;
    bwd_data_perf.resize(kRequestedBackwardDataAlgoCount);
    CUDNN_CHECK(cudnnFindConvolutionBackwardDataAlgorithm(handle_[0],
                                                          filter_desc_,
                                                          top_descs_[i],
                                                          conv_descs_[i],
                                                          bottom_descs_[i],
                                                          kRequestedBackwardDataAlgoCount,
                                                          &returnedAlgoCount,
                                                          &bwd_data_perf[0]));
    layer_perf_.bwd_data_perf[i] = vector<cudnnConvolutionBwdDataAlgoPerf_t>(bwd_data_perf.begin(),
                                                                             bwd_data_perf.begin() + returnedAlgoCount);

    need_optimize_ = true;
  }


  // Tensor descriptor for bias.
  if (this->bias_term_) {
    cudnn::setTensor4dDesc<Dtype>(&bias_desc_,
        1, this->num_output_ / this->group_, 1, 1);
  }
}

template<typename Dtype>
void CuDNNConvolutionLayer<Dtype>::AdjustWorkSpaces() {

  for (int x = 0; x < layer_perf_.fwd_algo.size(); ++x){
    cudnnConvolutionFwdAlgo_t new_algo = layer_perf_.fwd_perf[x][layer_perf_.fwd_algo[x]].algo;
    size_t new_mem = layer_perf_.fwd_perf[x][layer_perf_.fwd_algo[x]].memory;
    if ((new_algo != fwd_algo_[x]) || (new_mem != workspace_fwd_sizes_[x])) {
      fwd_algo_[x] = new_algo;
      workspace_fwd_sizes_[x] = layer_perf_.fwd_perf[x][layer_perf_.fwd_algo[x]].memory;
      if(workspace_fwd_sizes_[x] > workspaceData_fwd.size()){
        for (int g = 0; g < this->group_; ++g){
          workspaceData_fwd[g].reset(new SyncedMemory(workspace_fwd_sizes_[x]));
        }
      }
    }
  }

  for (int x = 0; x < layer_perf_.bwd_filter_algo.size(); ++x){
    cudnnConvolutionBwdFilterAlgo_t new_algo = layer_perf_.bwd_filter_perf[x][layer_perf_.bwd_filter_algo[x]].algo;
    size_t new_mem = layer_perf_.bwd_filter_perf[x][layer_perf_.bwd_filter_algo[x]].memory;

    if ((new_algo != bwd_filter_algo_[x]) || (new_mem != workspace_bwd_filter_sizes_[x])) {
      bwd_filter_algo_[x] = new_algo;
      workspace_bwd_filter_sizes_[x] = new_mem;
      if(workspace_bwd_filter_sizes_[x] > workspaceData_bwd_filter[0]->size()){
        for (int g = 0; g < this->group_; ++g){
          workspaceData_bwd_filter[g].reset(new SyncedMemory(new_mem));
        }
      }
    }
  }

  for (int x = 0; x < layer_perf_.bwd_data_algo.size(); ++x){
    cudnnConvolutionBwdDataAlgo_t new_algo = layer_perf_.bwd_data_perf[x][layer_perf_.bwd_data_algo[x]].algo;
    size_t new_mem = layer_perf_.bwd_data_perf[x][layer_perf_.bwd_data_algo[x]].memory;
    if ((new_algo != bwd_data_algo_[x]) || (new_mem != workspace_bwd_data_sizes_[x])) {
      bwd_data_algo_[x] = new_algo;
      workspace_bwd_data_sizes_[x] = new_mem;
      if(workspace_bwd_data_sizes_[x] > workspaceData_bwd_data[0]->size()){
        for (int g = 0; g < this->group_; ++g){
          workspaceData_bwd_data[g].reset(new SyncedMemory(new_mem));
        }
      }
    }
  }


}

template <typename Dtype>
CuDNNConvolutionLayer<Dtype>::~CuDNNConvolutionLayer() {
  // Check that handles have been setup before destroying.
  if (!handles_setup_) { return; }

  for (int i = 0; i < bottom_descs_.size(); i++) {
    cudnnDestroyTensorDescriptor(bottom_descs_[i]);
    cudnnDestroyTensorDescriptor(top_descs_[i]);
    cudnnDestroyConvolutionDescriptor(conv_descs_[i]);
  }
  if (this->bias_term_) {
    cudnnDestroyTensorDescriptor(bias_desc_);
  }
  cudnnDestroyFilterDescriptor(filter_desc_);

  int total_stream_per_group = CUDNN_FWD_STREAMS_PER_GROUP + CUDNN_BWD_STREAMS_PER_GROUP;
  for (int g = 0; g < this->group_ * total_stream_per_group; g++) {
    cudaStreamDestroy(stream_[g]);
    cudnnDestroy(handle_[g]);
  }

  // release all allocated workspace memory blocks.
  workspaceData_bwd_filter.empty();
  workspaceData_bwd_data.empty();
  workspaceData_fwd.empty();

  // unregister the layer perf
  typename boost::unordered_map<CuDNNConvolutionLayer*, PerfReg*>::iterator
          it = perf_reg.find(this);
  if (it != perf_reg.end()){
    perf_reg.erase(it);
  }

  delete [] stream_;
  delete [] handle_;
  delete [] fwd_algo_;
  delete [] bwd_filter_algo_;
  delete [] bwd_data_algo_;
  delete [] workspace_fwd_sizes_;
  delete [] workspace_bwd_data_sizes_;
  delete [] workspace_bwd_filter_sizes_;

  if (perf_reg.find(this) != perf_reg.end()){
    // un-register when the layer gets destroyed
    perf_reg.erase(this);
  }
}

INSTANTIATE_CLASS(CuDNNConvolutionLayer);

}   // namespace caffe
#endif
