
import math
# python中的闭包
# python中的装饰器


def func_out(num1):     # 该方法返回的是一个function
    def func_in(num2):
        return num1 + num2
    return func_in


def log(func):
    def wrapper(stars):
        func("who is I", stars)
        func("aaa", stars)
        func("I Im ron man", stars)
    return wrapper


# 闭包方式实现两点之间的距离
def get_dis_out(x1, y1):
    def get_dis_in(x2, y2):
        return math.sqrt((x1-x2)**2 + (y1-y2)**2)
    return get_dis_in


def logs(func):
    def wra():
        print('开始了')
        func()
        print('完了')
    return wra


@logs    # 把这个function传入 logs方法的参数内
def my_deca():
    print("よかった")


if __name__ == '__main__':
    f = func_out(20)
    print('结果:', f(30))
    # 闭包
    fs = log(print)
    fs('呵呵')
    # 改行是上两行的缩写
    log(print)('哈哈')

    # 求点(1,1)距离原点(0, 0)的距离
    result = get_dis_out(0, 0)  # 因为原点0,0的参实在调用函数的时候就已经实现写入了，所以闭包只需要传入变动的参数
    print('1,1距离原点的位置是:', result(1, 1))
    # 求点(2,1)距离原点(0, 0)的距离
    print('1,1距离原点的位置是:', result(2, 1))
    # 求点(3,3)距离原点(0, 0)的距离
    print('1,1距离原点的位置是:', result(3, 3))
    # 执行这个方法会执行在my_deca上的@的logs方法，my_deca会把自己当作参数传入logs方法，并运行logs方法
    my_deca()