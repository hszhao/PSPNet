## DenseCRF 

### Code

The code is modified from the publicly available code by Philipp Krähenbühl and Vladlen Koltun.
See their project [website](http://www.philkr.net/home/densecrf) for more information 

If you also use this part of code, please cite their [paper](http://googledrive.com/host/0B6qziMs8hVGieFg0UzE0WmZaOW8/papers/densecrf.pdf):
Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials, Philipp Krähenbühl and Vladlen Koltun, NIPS 2011.

### How to compile the code

Linux: 

  Run make command (modify Makefile if needed).

Please see run_densecrf.sh for examples of input arguments or see the dense_inference.cpp.

### Caffe wrapper

We have also provided a wrapper for Philipp's implementation in Caffe (see the layer densecrf_layer.cpp)