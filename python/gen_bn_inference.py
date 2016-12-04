import numpy as np
import sys
import os
import os.path as osp
import google.protobuf as pb
from argparse import ArgumentParser

pycaffe_dir = osp.dirname(__file__)
if osp.join(pycaffe_dir) not in sys.path:
    sys.path.insert(0, pycaffe_dir)
import caffe
from caffe.proto import caffe_pb2


def update_blob_name(blobs, old, new):
    if old not in blobs: return
    names = list(blobs)
    names[names.index(old)] = new
    del(blobs[:])
    blobs.extend(names)


def check(old_net, new_net, input_name='data'):
    caffe.set_device(0)
    caffe.set_mode_gpu()
    inputs = np.random.rand(*old_net.blobs[input_name].data.shape)
    inputs = inputs.astype(np.float32)
    old_net.blobs[input_name].data[...] = inputs
    new_net.blobs[input_name].data[...] = inputs
    ans = old_net.forward()
    out = new_net.forward()
    for k in ans:
        assert np.allclose(ans[k], out[k]), "Conversion failed"


def main(args):
    # Set default output file names
    if args.output_model is None:
        file_name = osp.splitext(args.model)[0]
        args.output_model = file_name + '_inference.prototxt'
    if args.output_weights is None:
        file_name = osp.splitext(args.weights)[0]
        args.output_weights = file_name + '_inference.caffemodel'
    with open(args.model) as f:
        model = caffe_pb2.NetParameter()
        pb.text_format.Parse(f.read(), model)

    # Determince the BN layers to be absorbed or replaced
    # Create the new layers
    new_layers = []
    absorbed, replaced = {}, {}
    for i, layer in enumerate(model.layer):
        if layer.type != 'BN':
            new_layers.append(layer)
            continue
        assert len(layer.bottom) == 1
        assert len(layer.top) == 1
        bottom_blob = layer.bottom[0]
        top_blob = layer.top[0]
        # Check if can be absorbed. As there could be some inplace layers,
        # for example, conv -> relu -> bn. In such case, the BN cannot be
        # absorbed.
        can_be_absorbed = False
        for j in xrange(i - 1, -1, -1):
            if bottom_blob in model.layer[j].top:
                if model.layer[j].type not in ['Convolution', 'InnerProduct']:
                    can_be_absorbed = False
                    break
                else:
                    can_be_absorbed = True
                    bottom_layer = model.layer[j]
        if can_be_absorbed:
            # Rename the blob in the top layers
            for j in xrange(i + 1, len(model.layer)):
                update_blob_name(model.layer[j].bottom, top_blob, bottom_blob)
                update_blob_name(model.layer[j].top, top_blob, bottom_blob)
            if bottom_layer.type == 'Convolution':
                bottom_layer.convolution_param.bias_term = True
            elif bottom_layer.type == 'InnerProduct':
                bottom_layer.inner_product_param.bias_term = True
            absorbed[layer.name] = bottom_layer.name
        elif args.replace_by == 'affine':
            # Replace by an scale bias layer
            new_layer = caffe_pb2.LayerParameter()
            new_layer.name = layer.name + '_affine'
            new_layer.type = 'Scale'
            new_layer.bottom.extend([bottom_blob])
            new_layer.top.extend([top_blob])
            new_layer.scale_param.bias_term = True
            replaced[layer.name] = new_layer.name
            new_layers.append(new_layer)
        elif args.replace_by == 'frozen':
            # Freeze the BN layer
            layer.bn_param.frozen = True
            del(layer.param[:])
            param = caffe_pb2.ParamSpec()
            param.lr_mult = 0
            param.decay_mult = 0
            layer.param.extend([param] * 2)
            new_layers.append(layer)

    # Save the prototxt
    output_model = caffe_pb2.NetParameter()
    output_model.CopyFrom(model)
    del(output_model.layer[:])
    output_model.layer.extend(new_layers)
    with open(args.output_model, 'w') as f:
        f.write(pb.text_format.MessageToString(output_model))

    # Copy the parameters
    weights = caffe.Net(args.model, args.weights, caffe.TEST)
    output_weights = caffe.Net(args.output_model, caffe.TEST)
    for name in np.intersect1d(weights.params.keys(),
                               output_weights.params.keys()):
        # Some original conv / inner product layers do not have bias_term
        for i in xrange(min(len(weights.params[name]),
                            len(output_weights.params[name]))):
            output_weights.params[name][i].data[...] = \
                weights.params[name][i].data.copy()

    # Absorb the BN parameters
    for old, new in absorbed.iteritems():
        scale, bias, mean, tmp = [p.data.ravel() for p in weights.params[old]]
        invstd = tmp if args.bn_style == 'invstd' else \
                 np.power(tmp + args.epsilon, -0.5)
        W, b = output_weights.params[new]
        assert W.data.ndim == 4 or W.data.ndim == 2
        assert b.data.ndim == 1
        if W.data.ndim == 4:
            W.data[...] = (W.data * scale[:, None, None, None]
                                  * invstd[:, None, None, None])
        elif W.data.ndim == 2:
            W.data[...] = W.data * scale[:, None] * invstd[:, None]
        b.data[...] = (b.data[...] - mean) * scale * invstd + bias

    # Fill up the affine layers
    for old, new in replaced.iteritems():
        scale, bias, mean, tmp = [p.data.ravel() for p in weights.params[old]]
        invstd = tmp if args.bn_style == 'invstd' else \
                 np.power(tmp + args.epsilon, -0.5)
        W, b = output_weights.params[new]
        assert W.data.ndim == 1
        assert b.data.ndim == 1
        W.data[...] = scale * invstd
        b.data[...] = bias - scale * mean * invstd

    # Check if the conversion is correct
    check(weights, output_weights)

    # Save the caffemodel
    output_weights.save(args.output_weights)


if __name__ == '__main__':
    parser = ArgumentParser(
            description="Generate Batch Normalized model for inference")
    parser.add_argument('model', help="The net definition prototxt")
    parser.add_argument('weights', help="The weights caffemodel")
    parser.add_argument('--output_model')
    parser.add_argument('--output_weights')
    parser.add_argument('--bn_style', type=str, default='var',
                        choices=['var', 'invstd'])
    parser.add_argument('--epsilon', type=float, default=1e-5,
                        help="The epsilon only used when bn_style is 'var'")
    parser.add_argument('--replace_by', type=str, default='affine',
                        choices=['affine', 'frozen'],
                        help="When a BN layer cannot be absorbed, replace it "
                             "by either affine (scale + bias) layers or a "
                             "frozen BN layer")
    args = parser.parse_args()
    main(args)
