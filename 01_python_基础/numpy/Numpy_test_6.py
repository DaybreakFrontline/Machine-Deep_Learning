# 数组的分割
import numpy as np


def split_te():
    x = np.arange(1, 9)
    # np.split(ary, indices_or_sections, axis)
    # ary:目标数组  indices_or_sections:如果是整数，就用平均切分，如果是数组，就延轴切分
    # axis: 沿着那个维度进行切人，默认为0，横向， 1的时候为纵向
    print(x)
    x2 = np.split(x, 4)     # 切割成四份
    print(x2)

    x3 = np.split(x, [3, 5])
    # [1 2 3 4 5 6 7 8] , 从下标3开始切出一个数组，再从下标5切出一个数组
    # [array([1, 2, 3]), array([4, 5]), array([6, 7, 8])]
    print(x3)

    grid = np.arange(16).reshape(4, 4)
    a, b = np.split(grid, 2)    # 横向切割， 切成两份
    print(a)
    print(b)

    x3 = x.reshape(2, 4)    # 两行四列
    print(x3)
    t = np.transpose(x3)    # 转换为四行两列
    print(t)


if __name__ == '__main__':
    split_te()

