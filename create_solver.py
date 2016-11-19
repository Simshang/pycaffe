# -*- coding: utf-8 -*-

path = '/root/caffe/examples/cifar10/'
solver_file=path+'solver_shang.prototxt'     #solver文件保存位置

sp={}
sp['train_net']='"'+path+'train.prototxt"'  # 训练配置文件
sp['test_net']='"'+path+'val.prototxt"'     # 测试配置文件
sp['test_iter']='313'                  # 测试迭代次数
sp['test_interval']='782'              # 测试间隔
sp['base_lr']='0.001'                  # 基础学习率
sp['display']='782'                    # 屏幕日志显示间隔
sp['max_iter']='78200'                 # 最大迭代次数
sp['lr_policy']='"step"'                 # 学习率变化规律
sp['gamma']='0.1'                      # 学习率变化指数
sp['momentum']='0.9'                   # 动量
sp['weight_decay']='0.0005'            # 权值衰减
sp['stepsize']='26067'                 # 学习率变化频率
sp['snapshot']='7820'                   # 保存model间隔
sp['snapshot_prefix']='"snapshot"'    # 保存的model前缀
sp['solver_mode']='GPU'                # 是否使用gpu
sp['solver_type']='SGD'                # 优化算法

def write_solver():
    #写入文件
    with open(solver_file, 'w') as f:
        for key, value in sorted(sp.items()):
            if not(type(value) is str):
                raise TypeError('All solver parameters must be strings')
            f.write('%s: %s\n' % (key, value))
if __name__ == '__main__':
    write_solver()