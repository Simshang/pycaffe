# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import sys,os,caffe

caffe_root = './'
sys.path.insert(0, caffe_root + 'python')
os.chdir(caffe_root)
if not os.path.isfile(caffe_root + 'cifar10_quick_iter_4000.caffemodel'):
    print "caffemodel is not exist..."
else:
    print "Ready to Go ..."

caffe.set_mode_gpu()

net = caffe.Net(caffe_root + 'cifar10_quick.prototxt',
                caffe_root + 'cifar10_quick_iter_4000.caffemodel',
                caffe.TEST)

print str(net.blobs['data'].data.shape)

#加载测试图片，并显示
img = caffe.io.load_image('./dog4.png')
print img.shape
plt.imshow(img)
plt.axis('off')
print img.shape


