import caffe
import numpy as np
from VBMF import EVBMF


def conv_layer(name, num_output, kernel_size=1, pad=0, stride=1):
    layer = caffe.proto.caffe_pb2.LayerParameter()
    layer.type = 'Convolution'
    layer.name = name
    
    layer.convolution_param.num_output = num_output
    layer.convolution_param.kernel_size.append(kernel_size)
    layer.convolution_param.pad.append(pad)
    layer.convolution_param.stride.append(stride)
    
    return layer

def decompose_layer(layer, rank):
    param = layer.convolution_param
    name = [layer.name+'_S', layer.name+'_core', layer.name+'_T']
    
    num_output = param.num_output
    kernel_size = param.kernel_size
    pad = param.pad if len(param.pad)!=0 else [0]
    stride = param.stride if len(param.stride)!=0 else [1]
    
    decomposed_layer = [
        conv_layer(name[0], rank[1]),
        conv_layer(name[1], rank[0], kernel_size[0], pad[0], stride[0]),
        conv_layer(name[2], num_output),
    ]
    
    return decomposed_layer


#for resnet 50
def rename_nodes(model_def):
    layer_index = len(model_def.layer)
    for i in range(layer_index):
        t_layer = model_def.layer[i]
        if t_layer.type=='Convolution' and t_layer.name.find('branch2a')!=-1:
            bot_name = t_layer.name
        if t_layer.name.find('branch2b')!=-1:
            if t_layer.type in ['ReLU','BatchNorm','Scale']:
                t_layer.bottom[0] = bot_name[:-1]+'b_T'
                t_layer.top[0] = t_layer.bottom[0]
            if t_layer.type == 'Convolution':
                if t_layer.name[-1]=='S':
                    t_layer.bottom.extend([bot_name])
                    t_layer.top.extend([t_layer.name])
                else:
                    t_layer.bottom.extend([model_def.layer[i-1].name])
                    t_layer.top.extend([t_layer.name])

        if t_layer.type=='Convolution' and t_layer.name.find('branch2c')!=-1:
            t_layer.bottom[0] = bot_name[:-1]+'b_T'

        print t_layer


    return model_def

def estimate_ranks(weights):
    T0 = np.reshape(np.moveaxis(weights, 0, 0), (weights.shape[0], -1))
    T1 = np.reshape(np.moveaxis(weights, 1, 0), (weights.shape[1], -1))
    _, T0_rank, _, _ = EVBMF(T0)
    _, T1_rank, _, _ = EVBMF(T1)
    
    return [T0_rank.shape[0], T1_rank.shape[1]]
