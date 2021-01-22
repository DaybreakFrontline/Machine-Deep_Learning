
a = 1


def my_test():
    print('my function')


def print_info(age, name):      # 形参
    print('我叫{0}，今年{1}岁 , 是一名'.format(name, age))
    print(a)


def print_yasaka(name, age):
    print('my name is %s , my age is %d' % (name, age))     # %s = String %d = double %f = float


def func1():
    global a    # global会在方法内把全局变量给改了
    a = 20
    print(a)


def func2(cs, fd=10, ss=20, *hobby):    # 一个*是元组
    print(cs, fd, ss, hobby)


def func3(cs, fd=10, ss=20, **hobby):   # 两个*是字典
    print(cs, fd, ss, hobby)


if __name__ == '__main__':
    func2(12, 22, 33, "score", "count", "skile")
    func3(12, 22, 33, score="score", count="count", skile="skile")
    print(sum(range(1, 101)))
