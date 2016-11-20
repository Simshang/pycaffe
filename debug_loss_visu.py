# -*- coding: utf-8 -*-
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig

x = np.linspace(-4, 4, 30)
y = np.sin(x);

_, ax1 = plt.subplots()
ax2 = ax1.twinx()

# train loss -> 绿色
ax1.plot(x,y, 'g')
# test loss -> 黄色
ax1.plot(x,y, 'y')
# test accuracy -> 红色
ax2.plot(x,y, 'r')

ax1.set_xlabel('iteration')
ax1.set_ylabel('loss')
ax2.set_ylabel('accuracy')


savefig('./loss.jpg')