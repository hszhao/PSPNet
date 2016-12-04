#include <string>

#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/vision_layers.hpp"

#ifdef WITH_PYTHON_LAYER
#include "caffe/python_layer.hpp"
#endif

#ifdef USE_CUDNN
#include "caffe/util/cudnn.hpp"
#endif

namespace caffe {

// Get convolution layer according to engine.
template <typename Dtype>
shared_ptr<Layer<Dtype> > GetConvolutionLayer(
    const LayerParameter& param) {
  ConvolutionParameter_Engine engine = param.convolution_param().engine();
  if (engine == ConvolutionParameter_Engine_DEFAULT) {
    engine = ConvolutionParameter_Engine_CAFFE;
#ifdef USE_CUDNN
    engine = ConvolutionParameter_Engine_CUDNN;
#endif
  }
  if (engine == ConvolutionParameter_Engine_CAFFE) {
    return shared_ptr<Layer<Dtype> >(new ConvolutionLayer<Dtype>(param));
#ifdef USE_CUDNN
  } else if (engine == ConvolutionParameter_Engine_CUDNN) {
    return shared_ptr<Layer<Dtype> >(new CuDNNConvolutionLayer<Dtype>(param));
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
  }
}

REGISTER_LAYER_CREATOR(Convolution, GetConvolutionLayer);

// Get pooling layer according to engine.
template <typename Dtype>
shared_ptr<Layer<Dtype> > GetPoolingLayer(const LayerParameter& param) {
  PoolingParameter_Engine engine = param.pooling_param().engine();
  if (engine == PoolingParameter_Engine_DEFAULT) {
    engine = PoolingParameter_Engine_CAFFE;
#ifdef USE_CUDNN
    engine = PoolingParameter_Engine_CUDNN;
#endif
  }
  if (engine == PoolingParameter_Engine_CAFFE) {
    return shared_ptr<Layer<Dtype> >(new PoolingLayer<Dtype>(param));
#ifdef USE_CUDNN
  } else if (engine == PoolingParameter_Engine_CUDNN) {
    PoolingParameter p_param = param.pooling_param();
    if (p_param.pad() || p_param.pad_h() || p_param.pad_w() ||
        param.top_size() > 1) {
      LOG(INFO) << "CUDNN does not support padding or multiple tops. "
                << "Using Caffe's own pooling layer.";
      return shared_ptr<Layer<Dtype> >(new PoolingLayer<Dtype>(param));
    }
    return shared_ptr<Layer<Dtype> >(new CuDNNPoolingLayer<Dtype>(param));
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
  }
}

REGISTER_LAYER_CREATOR(Pooling, GetPoolingLayer);

// Get relu layer according to engine.
template <typename Dtype>
shared_ptr<Layer<Dtype> > GetReLULayer(const LayerParameter& param) {
  ReLUParameter_Engine engine = param.relu_param().engine();
  if (engine == ReLUParameter_Engine_DEFAULT) {
    engine = ReLUParameter_Engine_CAFFE;
#ifdef USE_CUDNN
    engine = ReLUParameter_Engine_CUDNN;
#endif
  }
  if (engine == ReLUParameter_Engine_CAFFE) {
    return shared_ptr<Layer<Dtype> >(new ReLULayer<Dtype>(param));
#ifdef USE_CUDNN
  } else if (engine == ReLUParameter_Engine_CUDNN) {
    return shared_ptr<Layer<Dtype> >(new CuDNNReLULayer<Dtype>(param));
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
  }
}

REGISTER_LAYER_CREATOR(ReLU, GetReLULayer);

// Get sigmoid layer according to engine.
template <typename Dtype>
shared_ptr<Layer<Dtype> > GetSigmoidLayer(const LayerParameter& param) {
  SigmoidParameter_Engine engine = param.sigmoid_param().engine();
  if (engine == SigmoidParameter_Engine_DEFAULT) {
    engine = SigmoidParameter_Engine_CAFFE;
#ifdef USE_CUDNN
    engine = SigmoidParameter_Engine_CUDNN;
#endif
  }
  if (engine == SigmoidParameter_Engine_CAFFE) {
    return shared_ptr<Layer<Dtype> >(new SigmoidLayer<Dtype>(param));
#ifdef USE_CUDNN
  } else if (engine == SigmoidParameter_Engine_CUDNN) {
    return shared_ptr<Layer<Dtype> >(new CuDNNSigmoidLayer<Dtype>(param));
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
  }
}

REGISTER_LAYER_CREATOR(Sigmoid, GetSigmoidLayer);

// Get softmax layer according to engine.
template <typename Dtype>
shared_ptr<Layer<Dtype> > GetSoftmaxLayer(const LayerParameter& param) {
  SoftmaxParameter_Engine engine = param.softmax_param().engine();
  if (engine == SoftmaxParameter_Engine_DEFAULT) {
    engine = SoftmaxParameter_Engine_CAFFE;
#ifdef USE_CUDNN
    engine = SoftmaxParameter_Engine_CUDNN;
#endif
  }
  if (engine == SoftmaxParameter_Engine_CAFFE) {
    return shared_ptr<Layer<Dtype> >(new SoftmaxLayer<Dtype>(param));
#ifdef USE_CUDNN
  } else if (engine == SoftmaxParameter_Engine_CUDNN) {
    return shared_ptr<Layer<Dtype> >(new CuDNNSoftmaxLayer<Dtype>(param));
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
  }
}

REGISTER_LAYER_CREATOR(Softmax, GetSoftmaxLayer);

// Get tanh layer according to engine.
template <typename Dtype>
shared_ptr<Layer<Dtype> > GetTanHLayer(const LayerParameter& param) {
  TanHParameter_Engine engine = param.tanh_param().engine();
  if (engine == TanHParameter_Engine_DEFAULT) {
    engine = TanHParameter_Engine_CAFFE;
#ifdef USE_CUDNN
    engine = TanHParameter_Engine_CUDNN;
#endif
  }
  if (engine == TanHParameter_Engine_CAFFE) {
    return shared_ptr<Layer<Dtype> >(new TanHLayer<Dtype>(param));
#ifdef USE_CUDNN
  } else if (engine == TanHParameter_Engine_CUDNN) {
    return shared_ptr<Layer<Dtype> >(new CuDNNTanHLayer<Dtype>(param));
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
  }
}

REGISTER_LAYER_CREATOR(TanH, GetTanHLayer);

// Get bn layer according to engine.
template <typename Dtype>
shared_ptr<Layer<Dtype> > GetBNLayer(const LayerParameter& param) {
  BNParameter_Engine engine = param.bn_param().engine();
  if (engine == BNParameter_Engine_DEFAULT) {
    engine = BNParameter_Engine_CAFFE;
#if defined(USE_CUDNN)
#if CUDNN_VERSION_MIN(5, 0, 0)
     engine = BNParameter_Engine_CUDNN;
#endif
#endif
  }
  if (engine == BNParameter_Engine_CAFFE) {
    LOG(INFO) << "Layer " << param.name() << " is using CAFFE engine.";
    return shared_ptr<Layer<Dtype> >(new BNLayer<Dtype>(param));
#if defined(USE_CUDNN)
#if CUDNN_VERSION_MIN(5, 0, 0)
  } else if (engine == BNParameter_Engine_CUDNN) {
    LOG(INFO) << "Layer " << param.name() << " is using CUDNN engine.";
    return shared_ptr<Layer<Dtype> >(new CuDNNBNLayer<Dtype>(param));
#endif
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " calls cuDNN engine, but cuDNN version higher than 5.0 is not found";
  }
}

REGISTER_LAYER_CREATOR(BN, GetBNLayer);

#ifdef WITH_PYTHON_LAYER
PyThreadState* tstate = NULL;
template <typename Dtype>
shared_ptr<Layer<Dtype> > GetPythonLayer(const LayerParameter& param) {
  boost::lock_guard<boost::mutex> lock(mtx_);
  if (Caffe::py_tstate() == NULL){
    Py_Initialize();
    PyEval_InitThreads();
    PyThreadState* tstate = PyEval_SaveThread();
    Caffe::set_py_tstate(tstate);
  }
  PyGILState_STATE state;
  state = PyGILState_Ensure();
  try {

    bp::object module = bp::import(param.python_param().module().c_str());
    bp::object layer = module.attr(param.python_param().layer().c_str())(param);
    return bp::extract<shared_ptr<PythonLayer<Dtype> > >(layer)();
  } catch (bp::error_already_set) {
    PyErr_Print();
    throw;
  }
  PyGILState_Release(state);
}

REGISTER_LAYER_CREATOR(Python, GetPythonLayer);

boost::mutex mtx_;
#endif

// Layers that use their constructor as their default creator should be
// registered in their corresponding cpp files. Do not register them here.
}  // namespace caffe
