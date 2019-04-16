class layer:
    # 定义layer类,以后所有网络层都会继承这个主类
    def __init__(self):
        pass
    def feedforward(self, data):
        return data
    def backforward(self, backdelta, lastweights):
        pass
    def train(self):
        pass
    def predict(self):
        pass

