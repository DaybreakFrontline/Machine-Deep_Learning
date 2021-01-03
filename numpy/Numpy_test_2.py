# numpy的随机数
import numpy as np


def random_generator():
    x = np.random.random(size=4)    # 0-1之间
    print(x)
    print('------------------------------------------------------------')
    y = np.random.random(size=(3, 4))   # 三行四列  # 0-1之间
    print(y)


def random_generator_int():
    x = np.random.randint(20, 50, size=3)   # 生成的是整数
    print(x)
    print('------------------------------------------------------------')
    y = np.random.randint(20, 50, size=(3, 6))  # 三行六列 20-50之间
    print(y)


def random_generator_normal():
    x = np.random.randn()   # randn normal 正态分布
    print(x)
    print('------------------------------------------------------------')
    y = np.random.randn(2, 4)
    print(y)
    print('------------------------------------------------------------')
    z = np.random.randn(2, 3, 4)
    print(z)


def random_seed():  # 随机数种子
    x = np.random.seed(42)
    y = np.random.randn()
    print(x, " ", y)


def random_shuffle():   # 洗牌，把序列进行随机排列
    arr = np.array(np.arange(1, 100))
    print(arr)          # 没有打乱前
    np.random.shuffle(arr)
    print(arr)          # 打乱之后


if __name__ == '__main__':
    random_generator()
    random_generator_int()
    random_generator_normal()
    random_seed()
    random_shuffle()
