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

def rename_nodes(model_def, new_layers):
    layer_index = len(model_def.layer)
    for i in range(layer_index):
        t_layer = model_def.layer[i]
        
        if t_layer.type == 'Pooling':
            s_bot_name = t_layer.name
        #print t_layer
        #Label Decomposed layers nodes - Resnet18 version
        elif t_layer.type in ['BatchNorm','Scale']:
            if t_layer.name.find('branch2')!= -1:
                t_layer.bottom[0] = 'res'+t_layer.name[t_layer.name.find('_')-2:]+'_T'
                t_layer.top[0] = 'res'+t_layer.name[t_layer.name.find('_')-2:]+'_T'
            #print t_layer
        elif t_layer.type == 'ReLU':
            if t_layer.name.find('branch2')!= -1:
                t_layer.bottom[0] = t_layer.bottom[0]+'_T'
                t_layer.top[0] = t_layer.top[0]+'_T'


        if 'branch2a' in t_layer.name:
            if t_layer.name[-1]=='S':
                t_layer.bottom.extend([s_bot_name])
                t_layer.top.extend([t_layer.name])
            elif t_layer.name[-4:] =='core':
                t_layer.bottom.extend([model_def.layer[i-1].name])
                t_layer.top.extend([t_layer.name])
            elif t_layer.name[-1] =='T':
                t_layer.bottom.extend([model_def.layer[i-1].name])
                t_layer.top.extend([t_layer.name])

        elif 'branch2b' in model_def.layer[i].name:
            if t_layer.name[-1]=='S':
                t_layer.bottom.extend([t_layer.name[:-4]+'2a_T'])
                t_layer.top.extend([t_layer.name])
            elif t_layer.name[-4:] =='core':
                t_layer.bottom.extend([model_def.layer[i-1].name])
                t_layer.top.extend([t_layer.name])
            elif t_layer.name[-1] =='T':
                t_layer.bottom.extend([model_def.layer[i-1].name])
                t_layer.top.extend([t_layer.name])

        elif t_layer.type == 'Eltwise':
            t_layer.bottom[1] = t_layer.name + '_branch2b_T'
            s_bot_name = t_layer.name


        print t_layer


        '''
        if model_def.layer[i].name in new_layers:
            print '[A]'
            if i == 0:
                model_def.layer[i].bottom.extend(['data'])
            elif model_def.layer[i-1].type in ['ReLU']:
                model_def.layer[i].bottom.extend([model_def.layer[i-2].name])
            elif model_def.layer[i-1].type in ['Convolution','Pooling']:
                model_def.layer[i].bottom.extend([model_def.layer[i-1].name])
                print model_def.layer[i].bottom[0]
            elif model_def.layer[i-1].type in ['Data', 'HDF5Data', 'ImageData']:
                model_def.layer[i].bottom.extend([model_def.layer[i-1].top[0]])
            model_def.layer[i].top.extend([model_def.layer[i].name])
        #Rename Convolution layers nodes
        elif model_def.layer[i].type == 'Convolution':
            print '[B]'
            print model_def.layer[i].bottom[0]
            if i == 0:
                model_def.layer[i].bottom[0] = 'data'
            elif model_def.layer[i-1].name in new_layers: #no ReLU after Conv
                model_def.layer[i].bottom[0] = model_def.layer[i-1].name
            elif model_def.layer[i-2].name in new_layers:
                model_def.layer[i].bottom[0] = model_def.layer[i-2].name
        #Rename ReLU layers nodes
        elif model_def.layer[i].type == 'ReLU':
            if model_def.layer[i-1].name in new_layers:
                model_def.layer[i].bottom[0] = model_def.layer[i-1].name
                model_def.layer[i].top[0] = model_def.layer[i-1].name
        #Rename Pooling layers nodes
        elif model_def.layer[i].type == 'Pooling':
            if model_def.layer[i-1].name in new_layers: #no ReLU after Conv
                model_def.layer[i].bottom[0] = model_def.layer[i-1].name
            elif model_def.layer[i-2].name in new_layers:
                model_def.layer[i].bottom[0] = model_def.layer[i-2].name
        '''
    return model_def

def estimate_ranks(weights):
    T0 = np.reshape(np.moveaxis(weights, 0, 0), (weights.shape[0], -1))
    T1 = np.reshape(np.moveaxis(weights, 1, 0), (weights.shape[1], -1))
    _, T0_rank, _, _ = EVBMF(T0)
    _, T1_rank, _, _ = EVBMF(T1)
    
    return [T0_rank.shape[0], T1_rank.shape[1]]
