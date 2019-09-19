import os
import urllib
import caffe
from collections import OrderedDict
import cnn_tucker as tucker

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
os.chdir(ROOT_DIR)


model = {
    'proto': 'models/resnet18-deploy.prototxt',
    'weights': 'models/resnet18.caffemodel',
}

net = caffe.Net(model['proto'], model['weights'], caffe.TEST)

#You can choose ranks arbitrarily



ranks4_1 = [200, 100]
#...based on some heuristic (e.g T/3 and S/3, where S == T == 256)
ranks4_2 = [170, 170]
#...or estimate via VBMF
ranks4_3 = tucker.utils.estimate_ranks(net.params['conv4_3'][0].data)

layer_ranks = OrderedDict([
    ('conv4_1', ranks4_1),
    ('conv4_2', ranks4_2),
    ('conv4_3', ranks4_3),
])

paths = tucker.decompose_model(model['def'], model['weights'], layer_ranks)

print("\nDecomposed models saved to %s" %os.path.join(ROOT_DIR,os.path.dirname(paths[0])))
