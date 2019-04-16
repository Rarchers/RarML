import numpy as np
from numpy import random
import scipy.special  # 这是S函数,也就是sigmoid函数的库

from Layer import layer
np.random.seed(0)


def relu(x):
    return (x > 0) * x
def relu2deriv(output):
    return output > 0

# DNN层,接受一个一维向量,输出一维向量,两个DNN层中,上一层输出个数一定要等于下一层输入个数
class DNN(layer):
    def __init__(self, inputnodes, hidden_size, outputnodes, learn_rate=0.2, activation_functon="relu"):
        # 每一层的输入节点和输出节点,
        self.inodes = inputnodes
        self.onodes = outputnodes
        # 设置激活函数,默认是sigmoid
        self.acfun = activation_functon
        # 权重设置
        self.weights_0_1 = 2*np.random.random((inputnodes, hidden_size))-1
        self.weights_1_2 = 2*np.random.random((hidden_size, outputnodes))-1
        # 设置偏移度,防止神经元直接死亡 暂时简化一下,等于0
        self.bias = 0
        # 设置学习率
        self.alpha = learn_rate
        pass
    def train(self, input, target, loop):
        # 多次训练
        for iteration in range(loop):
            print("训练进度: "+str(iteration+1)+"/"+str(loop))

            # 误差值
            layer_2_error = 0
            # 每一个数据集训练
            for i in range(len(input)):
                layer_0 = input[i:i + 1]
                # 正向传播,计算得到本次输出
                layer_1 = relu(np.dot(layer_0, self.weights_0_1))
                layer_2 = np.dot(layer_1, self.weights_1_2)
                # 更新误差值,便于输出
                layer_2_error += np.sum((layer_2 - target[i:i + 1]) ** 2)
                # 反向传播的求偏导数
                layer_2_delta = (layer_2 - target[i:i + 1])
                layer_1_delta = layer_2_delta.dot(self.weights_1_2.T) * relu2deriv(layer_1)
                # 更新权重
                self.weights_1_2 -= self.alpha * layer_1.T.dot(layer_2_delta)
                self.weights_0_1 -= self.alpha * layer_0.T.dot(layer_1_delta)

                # 输出进度:
                x = i
                print("数据集"+str(i)+"",end=" ")
                while x>=0:
                    print("====", end="")
                    x-=1
                print(">", end="")

                y = len(input) - i
                while y>1:
                    print("----", end="")
                    y -= 1
                print("    误差:" + str(layer_2_error))
            print()
    # 预测
    def predict(self, input):
        layer_1 = relu(np.dot(input, self.weights_0_1))
        layer_2 = np.dot(layer_1, self.weights_1_2)
        print("预测结果:"+str(layer_2))