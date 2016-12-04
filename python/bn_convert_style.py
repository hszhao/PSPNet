import numpy as np
import sys
import os
import os.path as osp
from argparse import ArgumentParser

pycaffe_dir = osp.dirname(__file__)
if osp.join(pycaffe_dir) not in sys.path:
    sys.path.insert(0, pycaffe_dir)
import caffe


def main(args):
    net = caffe.Net(args.model, args.weights, caffe.TEST)
    conversion = args.conversion
    eps = args.epsilon
    for name, param in net.params.iteritems():
        if name.endswith('_bn'):
            if conversion == 'var_to_inv_std':
                var = param[3].data
                inv_std = np.power(var + eps, -0.5)
                param[3].data[...] = inv_std
            elif conversion == 'inv_std_to_var':
                inv_std = param[3].data
                var = np.power(inv_std, -2) - eps
                param[3].data[...] = var
            else:
                raise ValueError("Unknown conversion {}".format(conversion))
    if args.output is None:
        name, ext = osp.splitext(args.weights)
        suffix = conversion.split('_to_')[-1]
        args.output = name + '_' + suffix + ext
    net.save(args.output)


if __name__ == '__main__':
    parser = ArgumentParser(
        description="This script converts between two styles of BN models. "
                    "Specifically, in history we have two versions of BN "
                    "implementation---one storing running variance, and"
                    "the other storing running inverse std.")
    parser.add_argument('model', help="The deploy prototxt")
    parser.add_argument('weights', help="The caffemodel")
    parser.add_argument('--output', '-o', help="Output caffemodel")
    parser.add_argument('--conversion', type=str, default="inv_std_to_var",
                        choices=['inv_std_to_var', 'var_to_inv_std'],
                        help='can be "var_to_inv_std" or "inv_std_to_var"')
    parser.add_argument('--epsilon', type=float, default=1e-5,
                        help='the epsilon in the inverse, default to 1e-5')
    args = parser.parse_args()
    main(args)
