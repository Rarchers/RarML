# -*- coding: utf-8 -*-
from numpy import random
from math import e
from Layer import layer

# DNN层的更新版本!!!!!!!
class neuron:
    def __init__(self, typeName, net=0, out=None, sigma=None, value=None):
        self.typeName = typeName  # 三种神经元，input output 和 hidden
        self.net = net  # hidden和ouput神经元有此属性，为神经元的输入
        self.out = out  # hidden和ouput神经元有此属性，为神经元的输出
        self.sigma = sigma  # hidden和output神经元有此属性，用于backward propagation
        self.value = value  # input和output神经元有此属性，保存用户输入值


class DNNpro(layer):
    def __init__(self, inputSize, outputSize, layerSizeList, eta=0.5):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.layerNum = len(layerSizeList)  # 隐藏层数
        self.layerSizeList = layerSizeList  # 储存每个隐藏层有几个神经元
        self.inputNeuralList = []
        self.outputNeuralList = []
        self.paraArrayList = []
        self.biasList = []
        self.layerList = []  # 储存各个隐藏层，隐藏层中储存该层中的神经元
        self.eta = eta  # 控制梯度下降速度的参数
        # 未给出系数矩阵初始值，则随机生成

        # input 与 第一个隐层间的系数矩阵
        firstArray = random.random(size=(self.inputSize, layerSizeList[0]))
        self.paraArrayList.append(firstArray)
        # 隐层之间的系数矩阵
        for i in range(self.layerNum - 1):
            middleArray = random.random(size=(layerSizeList[i], layerSizeList[i + 1]))
            self.paraArrayList.append(middleArray)
        # 最后一个隐层与output间的系数矩阵
        lastArray = random.random(size=(layerSizeList[self.layerNum - 1], self.outputSize))
        self.paraArrayList.append(lastArray)
        # 未给出各层的bias，则随机产生
        biasList = []
        for i in range(self.layerNum + 1):
            self.biasList.append(random.random())
        # # 生成输入神经元列表
        # for input in inputList:
        #     self.inputNeuralList.append(neuron(typeName='input', value=input))
        # 生成隐藏神经元列表
        for size in layerSizeList:
            layer = []
            for i in range(size):
                layer.append(neuron(typeName='hidden'))
            self.layerList.append(layer)
        # # 生层输出神经元列表
        # for output in outputList:
        #     self.outputNeuralList.append(neuron(typeName='output', value=output))
    # 激活函数
    def sigmoid(self, x):
        return 1 / (1 + e ** (-x))

    def refresh(self, inputs, outputs):
        # 刷新训练列表
        for input in inputs:
            self.inputNeuralList = []
            self.inputNeuralList.append(neuron(typeName='input', value=input))
        for output in outputs:
            self.outputNeuralList = []
            self.outputNeuralList.append(neuron(typeName='output', value=output))

    # 计算输出值和真实值的差的平方和,再除2
    def error(self):
        sum = 0;
        for i in range(self.outputSize): sum += (self.outputNeuralList[i].value - self.outputNeuralList[i].out) ** 2
        return sum / 2.0
    # 计算隐藏层和输出层中神经元的net和out
    def forwardPropagation(self):
        # 处理第一个隐层
        for i in range(len(self.layerList[0])):
            currNeuron = self.layerList[0][i]
            currNeuron.net = 0
            for j in range(self.inputSize):
                currNeuron.net += self.paraArrayList[0][j][i] * self.inputNeuralList[j].value
            currNeuron.net += self.biasList[0]
            currNeuron.out = self.sigmoid(currNeuron.net)
        # 处理其他隐层
        for i in range(1, self.layerNum):
            for j in range(len(self.layerList[i])):  # 对于该隐层的每个神经元
                currNeuron = self.layerList[i][j]
                currNeuron.net = 0
                for k in range(len(self.layerList[i - 1])):  # 对于其上一层的每个神经元
                    currNeuron.net += self.layerList[i - 1][k].out * self.paraArrayList[i][k][j]
                    currNeuron.out = self.sigmoid(currNeuron.net)
        # 处理输出层
        for i in range(self.outputSize):
            currNeuron = self.outputNeuralList[i]
            currNeuron.net = 0
            for j in range(len(self.layerList[self.layerNum - 1])):
                currNeuron.net += self.layerList[self.layerNum - 1][j].out * \
                                  self.paraArrayList[len(self.paraArrayList) - 1][j][i]
            currNeuron.net += self.biasList[len(self.biasList) - 1]
            currNeuron.out = self.sigmoid(currNeuron.net)
    # 反向传播算法

    def backwardPropagation(self):
        # 先正向传播，更新各个神经元的net和out
        self.forwardPropagation()
        # 计算输出层的sigma
        for neuron in self.outputNeuralList:
            neuron.sigma = -(neuron.value - neuron.out) * neuron.out * (1 - neuron.out)
        # 计算最后一个隐藏层的sigma
        lastHiddenLayer = self.layerList[len(self.layerList) - 1]
        for i in range(len(lastHiddenLayer)):
            sum = 0
            for j in range(len(self.outputNeuralList)):
                sigma = self.outputNeuralList[j].sigma
                w = self.paraArrayList[len(self.paraArrayList) - 1][i][j]
                sum += sigma * w
            lastHiddenLayer[i].sigma = sum * lastHiddenLayer[i].out * (1 - lastHiddenLayer[i].out)
        # 计算其他隐藏层的sigma
        for i in range(self.layerNum - 1):
            sum = 0
            for j in range(len(self.layerList[i])):
                for k in range(len(self.layerList[i + 1])):
                    sigma = self.layerList[i + 1][k].sigma
                    w = self.paraArrayList[i + 1][j][k]
                    sum += sigma * w
                self.layerList[i][j].sigma = sum * self.layerList[i][j].out * (1 - self.layerList[i][j].out)
        # 更新连接矩阵的系数
        # 最后一个隐层和输出层间的系数矩阵
        lastHiddenLayer = self.layerList[len(self.layerList) - 1]
        for i in range(len(lastHiddenLayer)):
            for j in range(self.outputSize):
                self.paraArrayList[len(self.paraArrayList) - 1][i][j] -= self.eta * self.outputNeuralList[j].sigma * \
                                                                       lastHiddenLayer[i].out

        # 隐层间的系数矩阵
        for i in range(self.layerNum - 1):
            for j in range(len(self.layerList[i])):
                for k in range(len(self.layerList[i + 1])):
                    self.paraArrayList[i + 1][j][k] -= self.eta * self.layerList[i][j].out * self.layerList[i + 1][
                        k].sigma
        # 输入层与第一个隐层间的系数矩阵
        for i in range(self.inputSize):
            for j in range(len(self.layerList[0])):
                self.paraArrayList[0][i][j] -= self.eta * self.inputNeuralList[i].value * self.layerList[0][j].sigma
    def train(self, round):
        for i in range(round):
            self.backwardPropagation()
            print('error:', self.error())
            print('predict value:')
            for output in self.outputNeuralList:
                print(output.out)
    def predict(self):
        self.forwardPropagation()
        for output in self.outputNeuralList:
            print(output.out)



