# Numpy 科学计算基础库 提供多维数组(nd_array)、向量处理等

# 通过本章
# 了解什么是Numpy
# 掌握如何使用Numpy操作数组
# 掌握Numpy的常用函数，如存取函数，加权函数，最大最小极值等操作
import numpy as np


def np_test1():
    ar = np.arange(10)
    # 返回数组的个数
    print(ar.shape)
    # 获取第一个维度的数值
    print(ar.shape[0])


def numpy(arrays):
    # arrays 数组或者嵌套数列，其余的都是可选项
    # dtype元素数据类型，copy是否需要复制，order创建数组样式,C为行方向，F为列方向，A为默认任意方向
    # subok 默认返回一个与父类类型一致的数组， ndmin指定生成数组的最小维度
    arrs = np.array(arrays, dtype=None, copy=True, order=None, subok=False, ndmin=0)
    return arrs


def numpy_float(arrays):
    # 转成浮点  dtype == dataType
    arrs = np.array(arrays, dtype=float)
    return arrs


if __name__ == '__main__':
    # 创建np数组
    arr = np.arange(10)
    print(arr)
    print(type(arr))
    np_test1()
    my_list = [1, 3, 4, 7, 3, 22, 54, 11, 85, 99]
    print(numpy(my_list))
    # 二维数组
    my_2dList = [[1, 4, 7], [2, 5, 8], [3, 6, 9]]
    np_2list = np.array(my_2dList)  # 数组转np数组
    print(len(np_2list.shape))  # 数组是几维的
    print(np_2list)
    # 转成浮点类型
    print(numpy_float(my_list))
    # np.arange()
    print(np.arange(10.5))
    # 从1到9 每次步进2，类型float
    print(np.arange(1, 10, 2, dtype=float))
    # np.arange() 返回的是个一维数组，我们可以试着联合使用
    b = np.array([np.arange(1, 11), np.arange(11, 21), np.arange(21, 31)])
    print(b)
    print('b的维度是:', len(b.shape), b.shape)
