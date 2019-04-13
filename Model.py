
# 模组,使用时创建模组对象,在模组对象中添加神经网络层
class Model:
    def __init__(self):
        # 初始化两个栈来保存层信息
        self.stack1 = Stack()
        self.stack2 = Stack()
        pass
    def add(self, layers):
        # 添加网络层
        self.stack1.push(layers)
        pass
    def train(self, input_list, target_value, train_loop):
        loop = 1
        while loop <= train_loop:
            print()
            print()
            print("当前训练进度:"+ str(loop)+"/"+str(train_loop))
            # 在这里进行数据的解析工作!!!!!!! 2019.4.13更新,可以使用矩阵来输入多组数据进行训练
            line = 0
            while (line<input_list.shape[0]):
                list = input_list[line]
                tlist = target_value[line]
                print(list)
                print(tlist)
                # 因为用的栈,所以这里需要再用一个栈来调整层的顺序,确保首次添加的是第一层
                while not self.stack1.is_empty():
                    self.stack2.push(self.stack1.pop())
                # 确认是不是输入层
                FeedTop = True
                # 保存每一层的数据,用于正向传递
                trainlist = []
                # 正向传递
                while not self.stack2.is_empty():
                    layer = self.stack2.pop()
                    if FeedTop .__eq__(True):
                        trainlist = layer.feedforward(list).T
                        FeedTop = False
                    else:
                        trainlist = layer.feedforward(trainlist).T
                    # print(trainlist)
                    self.stack1.push(layer)
                # 反向传播:
                layertotal = self.stack1.size()
                while not self.stack1.is_empty():
                    # 取出一个网络层
                    layer = self.stack1.pop()
                    print("层"+str(layertotal-self.stack1.size())+str(layer))
                    # 开始反向传播
                    layer.backforward(layer.getoutputs().T, tlist)
                    self.stack2.push(layer)
                    print()
                # 在反向传递之后,需要再次把stack1填满,以便下一次的训练
                while not self.stack2.is_empty():
                    self.stack1.push(self.stack2.pop())
                line += 1
            loop += 1

    def evaluate(self, input_list):
        # 调整顺序
        while not self.stack1.is_empty():
            self.stack2.push(self.stack1.pop())
            # 确认是不是输入层
        FeedTop = True
        # 保存每一层的数据,用于正向传递
        trainlist = []
        # 正向传递
        while not self.stack2.is_empty():
            layer = self.stack2.pop()
            if FeedTop.__eq__(True):
                trainlist = layer.feedforward(input_list).T
                FeedTop = False
            else:
                trainlist = layer.feedforward(trainlist).T
            # print(trainlist)
            self.stack1.push(layer)
        lastlayer = self.stack1.peek()
        print(lastlayer.getoutputs().T)



class Stack(object):
    """栈"""
    def __init__(self):
         self.items = []

    def is_empty(self):
        """判断是否为空"""
        return self.items == []

    def push(self, item):
        """加入元素"""
        self.items.append(item)

    def pop(self):
        """弹出元素"""
        return self.items.pop()

    def peek(self):
        """返回栈顶元素"""
        return self.items[len(self.items)-1]

    def size(self):
        """返回栈的大小"""
        return len(self.items)
