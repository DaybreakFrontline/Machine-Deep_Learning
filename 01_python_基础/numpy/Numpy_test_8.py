# numpy聚合函数
import numpy as np


def test():
    a = np.arange(1, 30, dtype=int)
    print(a)
    print('------------------------------------------------------')

    print(np.sum(a))        # 对所有元素加和
    print(np.product(a))    # 对所有元素相乘
    print(np.mean(a))       # 求平均值
    print(np.std(a))        # 求标准差
    print(np.power(a, 2))   # 幂运算， 每个元素的n次方

    print(np.sqrt(a))       # 开平方
    print(np.argmin(a))     # 最小值的下标
    print(np.argmax(a))     # 最大值的下标
    print(np.inf)           # 无穷大

    print(np.exp(10))       # 以e为底的指数 e^10  e约等于2.71828……,是一个无理数,它是(1+1/n)的n次方的极限(n趋向于无穷大)
    print(np.log(10))       # 对数    exp的反向  exp与log互为反函数


if __name__ == '__main__':
    test()
