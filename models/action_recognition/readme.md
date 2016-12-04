##Spatial net (RGB input) models:
| name | caffemodel | caffemodel_url | license | caffe_commit |
| --- | --- | --- | --- | --- |
| Spatial UCF101 Split1 | cuhk_action_spatial_vgg_16_split1.caffemodel | http://mmlab.siat.ac.cn/very_deep_two_stream_model/cuhk_action_spatial_vgg_16_split1.caffemodel | license: non-commercial | d26b3b8b8eec182a27ce9871752fedd374b63650 
| Spatial UCF101 Split2 | cuhk_action_spatial_vgg_16_split2.caffemodel | http://mmlab.siat.ac.cn/very_deep_two_stream_model/cuhk_action_spatial_vgg_16_split2.caffemodel | license: non-commercial | d26b3b8b8eec182a27ce9871752fedd374b63650
| Spatial UCF101 Split3 | cuhk_action_spatial_vgg_16_split3.caffemodel | http://mmlab.siat.ac.cn/very_deep_two_stream_model/cuhk_action_spatial_vgg_16_split3.caffemodel | license: non-commercial | d26b3b8b8eec182a27ce9871752fedd374b63650

##Temporal net (optical flow input) models:
| name | caffemodel | caffemodel_url | license | caffe_commit |
| --- | --- | --- | --- | --- |
| Temporal UCF101 Split1 | cuhk_action_temporal_vgg_16_split1.caffemodel | http://mmlab.siat.ac.cn/very_deep_two_stream_model/cuhk_action_temporal_vgg_16_split1.caffemodel | license: non-commercial | d26b3b8b8eec182a27ce9871752fedd374b63650 
| Temporal UCF101 Split2 | cuhk_action_temporal_vgg_16_split2.caffemodel | http://mmlab.siat.ac.cn/very_deep_two_stream_model/cuhk_action_temporal_vgg_16_split2.caffemodel | license: non-commercial | d26b3b8b8eec182a27ce9871752fedd374b63650
| Temporal UCF101 Split3 | cuhk_action_temporal_vgg_16_split3.caffemodel | http://mmlab.siat.ac.cn/very_deep_two_stream_model/cuhk_action_temporal_vgg_16_split3.caffemodel | license: non-commercial | d26b3b8b8eec182a27ce9871752fedd374b63650

These models are trained using the strategy described in 
the [Arxvi report](http://arxiv.org/abs/1507.02159). Model and training configurations are set according to the original report. 

The model parameters are initialized with the public available VGG-16 model and trained on the UCF-101 dataset. 
The modified initialization models are provided

[Spatial](http://mmlab.siat.ac.cn/pretrain/vgg_16_action_rgb_pretrain.caffemodel), [Temporal](http://mmlab.siat.ac.cn/pretrain/vgg_16_action_flow_pretrain.caffemodel).

The bundled models are the iteration 15,000 snapshots using corresponding solvers.

[Project page](http://personal.ie.cuhk.edu.hk/~xy012/others/action_recog/).

These models were trained by Limin Wang @wanglimin and Yuanjun Xiong @yjxiong.

----

**Note**:
The training model `prototxt` file contains `"Gather"` layers which only work properly with this fork when "USE_MPI" is on. It is also possible to train the model with official Caffe codebase. You may need to incorporate the `VideoDataLayer`, remove `Gather` layers and restore all blob names suffixed by "_local" to their original names with out the suffix.

## License

The models are released for non-commercial use.
