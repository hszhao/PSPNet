#!/usr/bin/env python
"""
Example usage: To average caffenet_iter_10000.caffemodel,
caffenet_iter_20000.caffemodel, and caffenet_iter_30000.caffemodel. Use command

python2 polyak_average.py caffenet_deploy.prototxt caffenet_polyak.caffemodel \
  --weight_prefix caffenet --iter_range "(10000,30001,10000)"
"""
import numpy as np
import os.path as osp
import sys
from argparse import ArgumentParser

pycaffe_dir = osp.dirname(__file__)
if osp.join(pycaffe_dir) not in sys.path:
    sys.path.insert(0, pycaffe_dir)
import caffe


def main(args):
    if args.weight_files is not None:
        weight_files = args.weight_files
    else:
        weight_files = [args.weight_prefix + '_iter_{}.caffemodel'.format(it)
                        for it in args.iter_range]
    assert len(weight_files) > 0, "Must have at least one caffemodel"
    net = caffe.Net(args.model, weight_files[0], caffe.TEST)
    count = {param_name: 1 for param_name in net.params.keys()}
    for weight_file in weight_files[1:]:
        tmp = caffe.Net(args.model, weight_file, caffe.TEST)
        for param_name in np.intersect1d(net.params.keys(), tmp.params.keys()):
            count[param_name] += 1
            for w, v in zip(net.params[param_name], tmp.params[param_name]):
                w.data[...] += v.data
    for param_name in net.params:
        if count[param_name] <= 1: continue
        for w in net.params[param_name]:
            w.data[...] /= count[param_name]
    net.save(args.output)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('model', help="Net definition prototxt")
    parser.add_argument('output', help="Path for saving the output")
    parser.add_argument('--weight_files', type=str, nargs='+',
        help="A list of caffemodels")
    parser.add_argument('--weight_prefix', help="Prefix of caffemodels")
    parser.add_argument('--iter_range',
        help="Iteration range complementary with the prefix. In the form of "
             "(begin, end, step), where begin is inclusive while end is "
             "exclusive.")
    args = parser.parse_args()
    if args.weight_files is None and \
            (args.weight_prefix is None or args.iter_range is None):
        raise ValueError("Must provider either weight files or weight prefix "
                         "and iter range.")
    if args.iter_range is not None:
        args.iter_range = eval('xrange' + args.iter_range)
    main(args)
