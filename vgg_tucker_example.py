import os
import urllib
import caffe
from collections import OrderedDict
import cnn_tucker as tucker

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
os.chdir(ROOT_DIR)

if not os.path.isfile('models/VGG_ILSVRC_16_layers_deploy.caffemodel'):
    caffemodel_url = 'http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel'
    print("Downloading VGG_ILSVRC_16_layers_deploy.caffemodel...")
    urllib.urlretrieve (caffemodel_url, 'models/VGG_ILSVRC_16_layers_deploy.caffemodel')

model = {
    'def': 'models/VGG_ILSVRC_16_layers_deploy.prototxt',
    'weights': 'models/VGG_ILSVRC_16_layers_deploy.caffemodel',
}

net = caffe.Net(model['def'], model['weights'], caffe.TEST)

layers = [
            'conv1_1','conv1_2','conv2_1','conv2_2',
            'conv3_1','conv3_2','conv3_3','conv4_1','conv4_2','conv4_3',
            'conv5_1','conv5_2','conv5_3'
        ]

ranks = OrderedDict()
for i in range(len(layers)):
    ranks[layers[i]] = tucker.utils.estimate_ranks(net.params[layers[i]][0].data)


paths = tucker.decompose_model(model['def'], model['weights'], ranks)

print("\nDecomposed models saved to %s" %os.path.join(ROOT_DIR,os.path.dirname(paths[0])))
