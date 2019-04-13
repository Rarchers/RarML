import numpy as np
from DNN import DNN
# 这个类用于测试网络层
from Model import Model

if __name__ == '__main__':
    # # 以后抽象一个对象来统一管理所有的层,实现feedforward和backforward
    # # 现在暂时手动实现
    #
    # # 构建隐藏层的第一层
    # network = FCN(3, 2)
    # # 构建第二层
    # net1 = FCN(2, 1)
    # # 先试一下feedforward,传入[1, 2, 5]  矩阵计算后是一个2*1的矩阵,我们需要转置
    # print(network.feedforward([1, 2, 5]).T)
    # # 测试反向传播,注意:这里的结果一定是传最后的结果,也就是最后一层的输出神经元个数相应维度的矩阵
    # # 根据矩阵计算,例如2*3的矩阵和3*1的矩阵相乘后是2*1的矩阵,我们下一个参数传入的是1*n的矩阵,因此需要进行转置操作
    # net1.backforward(net1.feedforward(network.feedforward([1, 2, 5]).T), ([3]))
    # net1.backforward(net1.feedforward(network.feedforward([1, 2, 5]).T), ([3]))
    # net1.backforward(net1.feedforward(network.feedforward([1, 2, 5]).T), ([3]))
    # # 测试复合网络层
    # print(net1.feedforward(network.feedforward([1, 2, 5]).T))

    # 创建训练数据集
    train = [
    [-2, -1],
    [25, 6],
    [17, 4],
    [-15, -6],
    [0, 0]]
    # 创建训练答案
    key = [[1], [0], [0], [1], [0]]
    # 把数据集化为矩阵,传入训练模型
    train = np.array(train)
    key = np.array(key)

    test = [25, 6]
    test2 = [-2, -1]


    # 初始化模组
    model = Model()
    # 添加两个全连接层
    model.add(DNN(2, 4))
    model.add(DNN(4, 1))
    # 开始训练模型,训练次数为2
    model.train(train, key, 2000)

    model.evaluate(test)
    model.evaluate(test2)

