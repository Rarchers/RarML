import numpy as np
from Layer import layer


# 卷积层,对图像的三维矩阵进行卷积操作
class CNN(layer):
    def __init__(self, img, conv_filter):
        self.img = img
        self.conv_filter = conv_filter
        pass


    def conv(img, conv_filter):
        # 检查图像通道的数量是否与过滤器深度匹配
        if len(img.shape) > 2 or len(conv_filter.shape) > 3:
            if img.shape[-1] != conv_filter.shape[-1]:
                print("错误：图像和过滤器中的通道数必须匹配")
        # 检查过滤器是否是方阵
        if conv_filter.shape[1] != conv_filter.shape[2]:
            print('错误：过滤器必须是方阵')
        # 检查过滤器大小是否是奇数
        if conv_filter.shape[1] % 2 == 0:
            print('错误：过滤器大小必须是奇数')
        # 定义一个空的特征图，用于保存过滤器与图像的卷积输出
        feature_maps = np.zeros((img.shape[0] - conv_filter.shape[1] + 1,
                                 img.shape[1] - conv_filter.shape[1] + 1,
                                 conv_filter.shape[0]))
        # 卷积操作
        for filter_num in range(conv_filter.shape[0]):
            print("Filter ", filter_num + 1)
            curr_filter = conv_filter[filter_num, :]

            # 检查单个过滤器是否有多个通道。如果有，那么每个通道将对图像进行卷积。所有卷积的结果加起来得到一个特征图。
            if len(curr_filter.shape) > 2:
                conv_map = conv_(img[:, :, 0], curr_filter[:, :, 0])
                for ch_num in range(1, curr_filter.shape[-1]):
                    conv_map = conv_map + conv_(img[:, :, ch_num], curr_filter[:, :, ch_num])
            else:
                conv_map = conv_(img, curr_filter)
            feature_maps[:, :, filter_num] = conv_map
        return feature_maps

def conv_(self):
    filter_size = self.conv_filter.shape[1]
    result = np.zeros((self.img.shape))
    # 循环遍历,进行卷积
    for r in np.uint16(np.arange(filter_size / 2.0, img.shape[0] - filter_size / 2.0 * 1)):
        for c in np.uint16(np.arange(filter_size / 2.0, img.shape[1] - filter_size / 2.0 * 1)):
            # 卷积区:
            curr_region = self.img[r - np.uint16(np.floor(filter_size / 2.0)):r + np.uint16(np.ceil(filter_size / 2.0)),
                          c - np.uint16(np.floor(filter_size / 2.0)):c + np.uint16(np.ceil(filter_size / 2.0))]
            # 卷积操作
            curr_result = curr_region * self.conv_filter
            conv_sum = np.sum(curr_result)
            # 将求和保存到特征图中
            result[r, c] = conv_sum

    # 裁剪结果矩阵的异常值
    final_result = result[np.uint16(filter_size / 2.0):result.shape[0] - np.uint16(filter_size / 2.0),
                   np.uint16(filter_size / 2.0):result.shape[1] - np.uint16(filter_size / 2.0)]
    return final_result