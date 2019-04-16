# 模组,使用时创建模组对象,在模组对象中添加神经网络层
from DNN import DNN
from Stack import Stack

class Model:
    def __init__(self, eta=0.5):
        # 初始化两个栈来保存层信息
        self.stack1 = Stack()
        self.stack2 = Stack()
        pass

    def add(self, layers):
        # 添加网络层
        self.stack1.push(layers)
        pass

    def train(self, input_list, target_value, train_loop):
        layer = self.stack1.pop()
        if isinstance(layer, DNN):
            print("DNN神经网络开始训练:")
            layer.train(input_list,target_value,train_loop)
        else:
            print("其他神经网络")
        self.stack1.push(layer)
        pass

    def predict(self, input_list):
        layer = self.stack1.pop()
        if isinstance(layer, DNN):
            layer.predict(input_list)
        pass



