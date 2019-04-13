import numpy as np
from numpy import random
import scipy.special  # 这是S函数,也就是sigmoid函数的库

from Layer import layer


"""
请注意!!!!!!!!!!这个DNN层已经被废弃!!!!!!!!!!!!!!!!!!! (主要是因为反向传播的计算有点问题,后期可能会修改成为Flatten后的处理层)
如需使用DNN层,请使用DNN_PRO网络层来代替本层!!!!!!!!!!!!
Rarcher  2018.4.13
"""


# DNN层,接受一个一维向量,输出一维向量,两个DNN层中,上一层输出个数一定要等于下一层输入个数
class DNN(layer):
    def __init__(self, inputnodes, outputnodes, learn_rate=0.01, activation_functon="sigmoid"):
        # 每一层的输入节点和输出节点,用于构造权重矩阵
        self.inodes = inputnodes
        self.onodes = outputnodes
        # 设置激活函数,默认是sigmoid
        self.acfun = activation_functon
        # 权重设置
        self.wights = np.random.rand(outputnodes, inputnodes)
        # 设置偏移度,防止神经元直接死亡
        self.bias = random.random()
        # 设置学习率
        self.lr = learn_rate
        pass

    def feedforward(self, input_list):
        # 调整输入为二维矩阵,并修改为转置矩阵(为了后边的矩阵乘法做计算准备)
        inputs = np.array(input_list, ndmin=2).T
        # 计算加权数据
        inside = np.dot(self.wights, inputs) + self.bias
        # 选择激活函数
        if self.acfun == "sigmoid":
            activation_function = lambda x: scipy.special.expit(x)
            # 保存以下当前层的输出,为后边的反向传播做准备
            self.outputs = activation_function(inside)
            return self.outputs
        elif self.acfun == "relu":
            self.outputs = inside if inside > 0 else 0
            return self.outputs

    def backforward(self, output_value, target_value):
        # 调整输入为二维矩阵,并修改为转置矩阵(为了后边的矩阵乘法做计算准备)
        output = np.array(output_value, ndmin=2).T
        target = np.array(target_value, ndmin=2).T
        # 开始计算和正确值之间的差距:

        """2018.4.13修改
        
        原反向传播算法:
        """
        output_error = target - output
        print(output_error)
        # 矩阵之间的更新权重的表达式: ΔW(j,k)=α * Ek * sigmoid(Ok)*(1-sigmoid(Ok))*Oj(T)
        # α:学习率  Ek:误差矩阵 Ok当前层的输出 Oj前一层的输出矩阵
        # 推导:
        # 偏导计算后可以得到    new W(j,k) = oldW(j,k) - α*(δE/δW(j,k))
        # 矩阵运算推导
        # ΔW1,1   ΔW1,1   ΔW1,1   ...             | E1*S1(1-S1) |
        # ΔW1,1   ΔW1,1   ΔW1,1   ...         =   | E1*S1(1-S1) |  *    | O1   O2   03|
        # ΔW1,1   ΔW1,1   ΔW1,1   ...             | E1*S1(1-S1) |
        #  ...      ...      ...                     上边是下一层的值         这是前一层的值

        # 更新权重
        # 权重的矩阵测试输出
        # 更新权重,利用矩阵计算权重的调整值(本质其实还是求偏导数的链式求导法则)
        self.wights += self.lr * np.dot((output_error * output * (1.0 - output)).T, self.outputs)
        # 更新偏移度
        self.bias += self.lr * (output_error * output * (1.0 - output))
        # print("更新后weight: "+str(self.wights))
        # print("更新后bias: "+str(self.bias))
        pass
    def getoutputs(self):
        return self.outputs


