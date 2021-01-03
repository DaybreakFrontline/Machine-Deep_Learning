# 算数函数
import numpy as np


def math_te():
    a = np.arange(9, dtype=np.float).reshape(3, 3)
    print(a)
    b = np.array([10, 10, 10])
    print(b)
    print('**********************************************************')
    print(np.add(a, b))

    q = np.arange(9).reshape(3, 3)
    w = np.arange(9).reshape(3, 3)
    print('---------------------------------------------------------')
    print(np.add(q, w))         # 相加
    print(np.subtract(q, w))    # 相减

    print(np.multiply(q, 10))   # 相乘
    print(np.multiply(q, w))    # 相乘
    print(np.divide(q, w))      # 相除

    s = np.array([0, 30, 45, 60, 90])
    print(np.sin((a * np.pi / 180)))    # sin
    print(np.cos((a * np.pi / 180)))    # cos
    print(np.tan((a * np.pi / 180)))    # tan

    print(np.around(4.5))   # 4.0
    print(np.around(4.6))   # 5.0

    print(np.floor(4.5))    # 向下取整  4.0
    print(np.ceil(4.5))     # 相上取整  5.0


if __name__ == '__main__':
    math_te()
