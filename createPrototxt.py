# -*- coding: utf-8 -*-

from caffe import layers as L, params as P, to_proto

path = '/home/xxx/data/'  # 保存数据和配置文件的路径
train_lmdb = path + 'train_db'  # 训练数据LMDB文件的位置
val_lmdb = path + 'val_db'  # 验证数据LMDB文件的位置
mean_file = path + 'mean.binaryproto'  # 均值文件的位置
train_proto = path + 'train.prototxt'  # 生成的训练配置文件保存的位置
val_proto = path + 'val.prototxt'  # 生成的验证配置文件保存的位置


# 编写一个函数，用于生成网络
def create_net(lmdb, batch_size, include_acc=False):
    # 创建第一层：数据层。向上传递两类数据：图片数据和对应的标签
    data, label = L.Data(source=lmdb, backend=P.Data.LMDB, batch_size=batch_size, ntop=2,
                         transform_param=dict(crop_size=40, mean_file=mean_file, mirror=True))
    # 创建第二屋：卷积层
    conv1 = L.Convolution(data, kernel_size=5, stride=1, num_output=16, pad=2, weight_filler=dict(type='xavier'))
    # 创建激活函数层
    relu1 = L.ReLU(conv1, in_place=True)
    # 创建池化层
    pool1 = L.Pooling(relu1, pool=P.Pooling.MAX, kernel_size=3, stride=2)
    conv2 = L.Convolution(pool1, kernel_size=3, stride=1, num_output=32, pad=1, weight_filler=dict(type='xavier'))
    relu2 = L.ReLU(conv2, in_place=True)
    pool2 = L.Pooling(relu2, pool=P.Pooling.MAX, kernel_size=3, stride=2)
    # 创建一个全连接层
    fc3 = L.InnerProduct(pool2, num_output=1024, weight_filler=dict(type='xavier'))
    relu3 = L.ReLU(fc3, in_place=True)
    # 创建一个dropout层
    drop3 = L.Dropout(relu3, in_place=True)
    fc4 = L.InnerProduct(drop3, num_output=10, weight_filler=dict(type='xavier'))
    # 创建一个softmax层
    loss = L.SoftmaxWithLoss(fc4, label)

    if include_acc:  # 在训练阶段，不需要accuracy层，但是在验证阶段，是需要的
        acc = L.Accuracy(fc4, label)
        return to_proto(loss, acc)
    else:
        return to_proto(loss)


def write_net():
    # 将以上的设置写入到prototxt文件
    with open(train_proto, 'w') as f:
        f.write(str(create_net(train_lmdb, batch_size=64)))

    # 写入配置文件
    with open(val_proto, 'w') as f:
        f.write(str(create_net(val_lmdb, batch_size=32, include_acc=True)))


if __name__ == '__main__':
    write_net()