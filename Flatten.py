import numpy
from Layer import layer

# 平化层,将多维矩阵变为一维,便于输入DNN层
class Flatten(layer):
    def __init__(self):
        pass
    def feedforward(self, input_list):
        pass
    def backforward(self, output_value, target_value, acfun):
        pass