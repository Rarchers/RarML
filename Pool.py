import numpy as np

from Layer import layer

# 池化层 在CNN层之后,用于缩小矩阵
class Pool(layer):
    def __init__(self):
        pass
    def feedforward(self, size=2, stride=2):
        pass
    def backforward(self, output_value, target_value, acfun):
        pass

    def pooling(feature_map, size=2, stride=2):
        # 定义池化操作的输出
        pool_out = np.zeros((np.uint16((feature_map.shape[0] - size + 1) / stride + 1),
                             np.uint16((feature_map.shape[1] - size + 1) / stride + 1),
                             feature_map.shape[-1]))

        for map_num in range(feature_map.shape[-1]):
            r2 = 0
            for r in np.arange(0, feature_map.shape[0] - size + 1, stride):
                c2 = 0
                for c in np.arange(0, feature_map.shape[1] - size + 1, stride):
                    pool_out[r2, c2, map_num] = np.max([feature_map[r: r + size, c: c + size, map_num]])
                    c2 = c2 + 1
                r2 = r2 + 1
        return pool_out