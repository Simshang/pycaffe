# -*- coding: utf-8 -*-

import caffe
import matplotlib
matplotlib.use('Agg')
from matplotlib.pyplot import plot,savefig
root = '/root/caffe/examples/'   #根目录
deploy=root + 'mnist/deploy.prototxt'    #deploy文件
caffe_model=root + 'mnist/lenet_iter_9380.caffemodel'   #训练好的 caffemodel
net = caffe.Net(deploy,caffe_model,caffe.TEST)   #加载model和network

[(k,v[0].data.shape) for k,v in net.params.items()]  #查看各层参数规模
w1=net.params['Convolution1'][0].data  #提取参数w
b1=net.params['Convolution1'][1].data  #提取参数b

net.forward()   #运行测试
[(k,v.data.shape) for k,v in net.blobs.items()]  #查看各层数据规模
fea=net.blobs['InnerProduct1'].data   #提取某层数据（特征）

print fea