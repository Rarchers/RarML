class layer:
    # 定义layer类,以后包括CNN层都会继承这个主类
    def __init__(self):
        pass
    def feedforward(self, input_list):
        return input_list

    def backforward(self, output_value, target_value, acfun):
        pass