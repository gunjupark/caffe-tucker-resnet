import os
import urllib
import caffe
from collections import OrderedDict
import cnn_tucker as tucker
import sys

sys.stdout = open('output_log.txt','w')

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
os.chdir(ROOT_DIR)


model = {
    'def': 'models/resnet18-deploy.prototxt',
    'weights': 'models/resnet-18.caffemodel',
}

net = caffe.Net(model['def'], model['weights'], caffe.TEST)

#You can choose ranks arbitrarily

decomp_layers = [
        'res2a_branch2a','res2a_branch2b','res2b_branch2a','res2b_branch2b',
        'res3a_branch2a','res3a_branch2b','res3b_branch2a','res3b_branch2b',
        'res4a_branch2a','res4a_branch2b','res4b_branch2a','res4b_branch2b',
        'res5a_branch2a','res5a_branch2b','res5b_branch2a','res5b_branch2b'
        ]


ranks = OrderedDict()

for i in range(len(decomp_layers)):
    ranks[decomp_layers[i]] = tucker.utils.estimate_ranks(
            net.params[decomp_layers[i]][0].data)


tucker.decompose_model(model['def'], model['weights'], ranks)

#print("\nDecomposed models saved to %s" %os.path.join(ROOT_DIR,os.path.dirname(paths[0])))
