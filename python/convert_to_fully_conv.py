import numpy as np
import os
import os.path as osp
import sys
import google.protobuf as pb
from argparse import ArgumentParser

pycaffe_dir = osp.dirname(__file__)
if osp.join(pycaffe_dir) not in sys.path:
    sys.path.insert(0, pycaffe_dir)
import caffe
from caffe.proto import caffe_pb2


def main(args):
    caffe.set_mode_cpu()
    fc_net = caffe.Net(args.model, args.weights, caffe.TEST)
    # make fully conv prototxt
    fc_proto = caffe_pb2.NetParameter()
    with open(args.model, 'r') as f:
        pb.text_format.Parse(f.read(), fc_proto)
    layers = []
    fc_to_conv_dic = {}
    for layer in fc_proto.layer:
        if layer.type != 'InnerProduct':
            layers.append(layer)
            continue
        new_ = caffe_pb2.LayerParameter()
        new_.name = layer.name + '_conv'
        fc_to_conv_dic[layer.name] = new_.name
        new_.type = 'Convolution'
        new_.bottom.extend(layer.bottom)
        new_.top.extend(layer.top)
        new_.convolution_param.num_output = layer.inner_product_param.num_output
        bottom_shape = fc_net.blobs[layer.bottom[0]].data.shape
        if len(bottom_shape) == 4:
            new_.convolution_param.kernel_h = bottom_shape[2]
            new_.convolution_param.kernel_w = bottom_shape[3]
        else:
            new_.convolution_param.kernel_size = 1
        layers.append(new_)
    conv_proto = caffe_pb2.NetParameter()
    conv_proto.CopyFrom(fc_proto)
    del(conv_proto.layer[:])
    conv_proto.layer.extend(layers)
    if args.save_model is None:
        name, ext = osp.splitext(args.model)
        args.save_model = name + '_fully_conv' + ext
    with open(args.save_model, 'w') as f:
        f.write(pb.text_format.MessageToString(conv_proto))
    # make fully conv parameters
    conv_net = caffe.Net(args.save_model, args.weights, caffe.TEST)
    for fc, conv in fc_to_conv_dic.iteritems():
        conv_net.params[conv][0].data.flat = fc_net.params[fc][0].data.flat
        conv_net.params[conv][1].data[...] = fc_net.params[fc][1].data
    if args.save_weights is None:
        name, ext = osp.splitext(args.weights)
        args.save_weights = name + '_fully_conv' + ext
    conv_net.save(args.save_weights)
    print args.model, args.weights


if __name__ == '__main__':
    parser = ArgumentParser(
        description="Convert fully connected layers to convolution layers"
    )
    parser.add_argument(
        'model',
        help="Path to input deploy prototxt"
    )
    parser.add_argument(
        'weights',
        help="Path to input caffemodel"
    )
    parser.add_argument(
        '--save_model',
        help="Path to output deploy prototxt"
    )
    parser.add_argument(
        '--save_weights',
        help="Path to output caffemodel"
    )
    args = parser.parse_args()
    main(args)