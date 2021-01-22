# 切片和索引
import numpy as np


def split_1():        # 这玩意叫切片
    x = np.arange(10)
    print('原数组:', x)
    y = x[2:7:2]    # 切割从下标2开始，到下标7， 每次步进2
    z = x[2:]       # 从下标2开始到结束
    print(y)
    print(z)


def split_2():              # arr.reshape(x, y)  x * y == len(arr) == true  否则报错！
    x = np.arange(1, 13)    # 包括1 不包括13， 包头不包尾 供12个元素
    print(x)
    a = x.reshape(4, 3)     # 转换为一个4行3列的二维数组
    # x.reshape(3, 3)       # 注意此行会报错，是因为一共12个元素不能被转换成3行3列
    print(a)
    print(a[2])             # 二维数组的第下标2行的数据(就是第三行)
    print(a[2][1])          # 取出第三行的第二个元素

    v1 = [a[1, 0], a[1, 1], a[2, 0], a[2, 1]]
    v2 = a[1:3, 1:3]
    print(v2)

    x2 = np.copy(a[1:3, 1:3])   # copy目标的元素
    print(x2)


if __name__ == '__main__':
    split_1()
    print('--------------------------------------------------------------------------------')
    split_2()
