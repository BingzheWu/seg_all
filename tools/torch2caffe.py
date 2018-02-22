import torch
import sys
sys.path.append('/home/bingzhe/tools/pytorch2caffe')
from torch.autograd import Variable
from models import mobilenet 
import os
from pytorch2caffe import pytorch2caffe, plot_graph
from prototxt import *
model = mobilenet.mobilenet_v1()
model.eval()
input_var = Variable(torch.rand(1, 3, 299, 299))
output_var = model(input_var)
output_dir = 'caffemodel_dir'
# plot graph to png
#plot_graph(output_var, os.path.join(output_dir, 'mobilenet_v1.dot'))
net_info = pytorch2prototxt(input_var, output_var)
save_prototxt(net_info, os.path.join(output_dir, 'mv1.prototxt'))
'''
pytorch2caffe(input_var, output_var, 
              os.path.join(output_dir, 'mv1-pytorch2caffe.prototxt'),
              os.path.join(output_dir, 'mv1-pytorch2caffe.caffemodel'))
'''