import numpy as np
from DNN import DNN
# 这个类用于测试网络层
from Model import Model

# 如何使用框架来创建一个神经网络:
#   1.使用如下语句创建一个model对象来组建你的神经网络:
#       model = Model()
#   2.添加一个神经网络层(暂时只支持添加DNN层),使用如下语句:
#       model.add(layer)      layer为相应的神经网络层
#   3.使用如下语句训练你的神经网络:
#       model.train(input, target, loop)
#   4.使用训练好的神经网络开始预测:
#       model.predict(input)

# 关于我可以使用的网络层(佛系更新):
#   1.DNN层:
#       使用方法为 model.add(DNN(input_size, hidden_size, output_size))
#       完整传参: (self, inputnodes, hidden_size, outputnodes, learn_rate, activation_functon):
#       inputnodes:                 输入神经元个数
#       hidden_size:                隐藏层神经元个数
#       outputnodes:                输出层神经元个数
#       learn_rate:                 学习速率    ->默认为0.2
#       activation_function:        激活函数    ->默认为ReLU函数(暂时只支持ReLU函数)
# ----------------------------------------------------------------------------------------------- #
#       注意::DNN层只能添加一层,实际训练也只会训练最新添加的DNN层
# ----------------------------------------------------------------------------------------------- #

if __name__ == '__main__':
    # 训练集
    train = np.array([[1,0,1],
                      [0,1,1],
                      [0,0,1],
                      [1,1,1]])
    key = np.array([[1,1,0,0]]).T
    # 新建一个模组
    model = Model()
    # 添加一个DNN层
    model.add(DNN(3, 4, 1))
    # 训练60次
    model.train(train, key, 60)
    # 进行一次预测
    model.predict([0,0,0])

