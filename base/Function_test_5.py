from functools import reduce

my_squ = [123, 5543, 56.23, -3.21e4]
# 高阶函数， lambda表达式


def my_test(num):   # 传入数组的时候 相乘结果等数输出两编原数组
    return num * 2


def my_test_2(num):   # 传入数组的时候 相乘结果等数输出两编原数组
    return num * 2 + 66


# 高阶函数
def convert(func, seq):     # func是一个函数方法
    print('转换序列中的数值，要他们统一为一样的类型')
    # 通过for循环将seq中的每个元素取出来，交给func的函数里边 因为传入的func是int, 所以func(eachNum) == int(eachNum)
    return [func(eachNum) for eachNum in seq]


# 这个函数如果是让reduce使用的话，就必须接收两个函数
def my_add(a, b):
    return a + b


# 返回值是true或false
def my_condition(x):
    return x % 2 == 0


if __name__ == '__main__':
    print(my_test(my_squ))
    print(convert(int, my_squ))
    print(convert(float, my_squ))
    # 此处属于高阶函数 将my_test当作参数传入convert中，convert中每一个eachNum 都会经过my_test函数的计算处理
    print((convert(my_test, my_squ)))
    print(list(map(my_test, my_squ)))   # 这个map其实和我手写的convert函数是一个意思
    # 使用lambda表达式来省略方法  list(map(my_test, my_squ)) == list(map(lambda num: num * 2, my_squ)
    print(list(map(lambda num: num * 2, my_squ)), 'lambda')
    print(list(map(my_test_2, my_squ)))

    # 第一个参数是一个函数， 第二个参数是一个序列，reduce会将序列中元素进行累积，累积的规则由第一个函数来解释
    va = reduce(my_add, list(range(1, 101)))    # 记得引包  发现这玩意可以当做Σ用
    vb = reduce(lambda x, y: x + y, list(range(1, 101)))    # 此处 my_add 函数更换为 lambda x, y: x + y
    print(va)
    print(vb)

    # filter()函数会保留进入my_condition方法返回值为true的元素，
    fl = list(filter(my_condition, [1, 2, 3, 4, 5, 6, 7, 8, 9]))
    print(fl)

    # sorted 入参序列，默认升序. 加上key=，key=是一个function，你可以定义排序的规则
    print(sorted([2, 6, -1, 10, 55, -22, 91, 33]))
    print(sorted([2, 6, -1, 10, 55, -22, 91, 33], key=abs))  # 此处注意，abs是绝对值，根据绝对值排序，但是输出依然是源序列，只是按照每个元素的绝对值排序

    # lambda 表达式 list(filter(my_condition, [1, 2, 3, 4, 5, 6, 7, 8, 9]))   my_condition = lambda x: x % 2 == 0 #
    print(list(filter(lambda x: x % 2 == 0, [1, 2, 3, 4, 5, 6, 7, 8, 9])))


