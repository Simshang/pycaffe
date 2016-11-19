# -*- coding: utf-8 -*-
import caffe
# python file under the /root/caffe/

path = '/root/caffe/examples/cifar10/'
caffe.set_mode_gpu()
caffe.set_device(0)
solver = caffe.SGDSolver(path + 'cifar10_quick_solver.prototxt')
solver.solve()