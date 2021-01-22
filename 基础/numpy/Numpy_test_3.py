# ndarray对象 N维数组对象ndarray
# ndarray 是用来存放同类型元素的多维数组
# ndarray 中的每个元素在内存中都有相同储存大小的区域
# ndarray 由以下内容组成：一个指向数据的指针，数据的类型或dtype，一个数组形状shape的元组
import numpy as np


def __arr_generator():
    x1 = np.random.randint(1, size=6)
    print(x1)
    x2 = np.random.randint(10, size=(3, 4))
    print(x2)
    x3 = np.random.randn(3, 4, 2)
    print(x3)

    print('---------------------------------------------------------------------')
    print('ndim 属性'.center(20, '*'))
    print('ndim:', x1.ndim, x2.ndim, x3.ndim)       # ndim 数组维度
    print('shape 属性'.center(20, '*'))
    print('shape:', x1.shape, x2.shape, x3.shape)   # shape n行m列
    print('dtype 属性'.center(20, '*'))
    print('dtype:', x1.dtype, x2.dtype, x3.dtype)   # 元素类型
    print('size 属性'.center(20, '*'))
    print('size:', x1.size, x2.size, x3.size)       # 元素总个数， shape中相乘的结果


def __zeros_generator():    # order ‘C’ 或 ‘F’ 代表行优先还是列优先
    z1 = np.zeros(5, dtype=int)     # zeros创建指定大小的数组，元素都用0填充
    print(z1)

    z2 = np.ones(5, dtype=int)      # ones，元素都用1填充
    print(z2)

    z3 = np.empty(5, dtype=int)     # empty，元素都用现有的内存中的数值，开辟空间，但不清空空间，显示的之前空间里的值
    print(z3)

    # 生成一个10开始，20结束，分成20份的数组
    z4 = np.linspace(10, 200, num=20, dtype=int)   # linspace 建立等差的一维数组 包括开始和结束值
    # endpoint 为true时，数列中包含stop值，反之不包含，默认为true
    # retstep 为true时，生成的数组中会显示间距，反之不显示
    print(z4)

    # log2为底，输出2的0-9次方
    # start=开始值，stop=结束值，num=元素个数，base=指定对数的底
    z5 = np.logspace(0, 9, 10, base=2, dtype=int)
    print(z5)

    arr = ([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    z6 = np.zeros_like(arr)     # 把已有的数组的元素替换为0  zeros == 0
    z7 = np.ones_like(arr)  # 把已有的数组的元素替换为1 ones == 1
    print(z6)
    print(z7)
    print(np.equal(z6, z7))


if __name__ == '__main__':
    __arr_generator()
    print('////////////////////////////////////////////////////////////')
    __zeros_generator()