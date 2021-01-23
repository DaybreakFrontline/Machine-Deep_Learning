# 改变数组的维度以及数组的拼接
import numpy as np


def change_1():
    a = np.arange(24)
    print(a)
    b = a.reshape(2, 3, 4)  # 改变为2个三行四列
    print(b)
    print(b.shape)
    print('--------------------------------------------------------------')
    s = b.ravel()   # 将多维数组改变为低维数组
    print(s)
    print('--------------------------------------------------------------')
    x = a.reshape(3, 8)
    print(x)
    x2 = x.flatten()    # flatten 压扁、压平  将二维数组压扁为一维数组
    print(x2)
    print('*******************************************************************')


def arr_hstack():   # 数组的拼接 horizontal n. 水平
    a = np.array([[1, 2, 3], [4, 5, 6]])
    print(a)
    b = np.array([['a', 'b', 'c'], ['d', 'e', 'f']])
    print(b)
    print('--------------------------------------------------------------')
    co = np.hstack([a, b])  # 按照行进行拼接
    print(co)

    cp = np.vstack([a, b])  # 按照列进行拼接
    print(cp)

    e = np.concatenate([a, b], axis=1)  # axis=1 == hstack   按照行进行拼接
    print(e)
    r = np.concatenate([a, b], axis=0)  # axis=0 == vstack  按照列进行拼接
    print(r)
    print('*******************************************************************')


def arr_there_stack():
    aa = np.arange(1, 37).reshape(3, 4, 3)
    print(aa)
    bb = np.arange(101, 137).reshape(3, 4, 3)
    print(bb)
    cc = np.concatenate([aa, bb], axis=0)   # 按照列进行拼接
    cc = np.concatenate([aa, bb], axis=1)  # 按照列进行拼接
    cc = np.concatenate([aa, bb], axis=2)  # 按照深度进行拼接
    print(cc)


if __name__ == '__main__':
    change_1()
    arr_hstack()
    arr_there_stack()
