import os

def ri():
    value = 0
    while value < 4:
        for i in range(5):
            print(value, end = " ")
        value += 1
        print()


def ni():
    j = 1
    while j <= 9:
        i = 1
        while i <= j:
            print('%dx%d=%d'%(j,i,j*i), end=' ')
            i += 1
        print('')
        j += 1


def va():
    for j in range(1,10):   # py中是 左闭右开
        for i in range(1,j+1):
            print('%dx%d=%d'%(j,i,j*i), end=' ')
        print()


def te():
    for i in range(10):
        if i % 2 == 0:
            continue
        print(i)


def sd():
    names = ["AA", "BB", "CC", "DD"]
    print(range(len(names)))
    for i in range(len(names)):
        if i >= 2:
            break
        print(names[i])


def qp():                   # 包前不包后 左闭右开
    pystring = "abcdefgh"
    print(pystring[:])      # abcdefg
    print(pystring[2:])     # cdefgh    从下标xx开始
    print(pystring[4:])     # efgh
    print(pystring[:2])     # ab        到下标xx-1截至
    print(pystring[2:4])    # cd    从下标2开始到下标4-1结束
    print(pystring[1:6:2])  # bcde -> bdf
    print(pystring[-3:])    # fgh
    print(pystring[5:])     # fgh
    print(pystring[-5:-3])  # de
    print(pystring[::-1])   # hgfedcba

def ty():
    aa = [1,2,3,4,1]
    bb = aa.copy()
    print(id(aa))
    print(id(bb))
    print(aa.count(1))

def so():
    a = [2,3,6,8,1,2,7,4,9]
    a.sort()    # 改变原有的数组顺序
    b = sorted(a)   # 不改变原有的数组顺序 生成新的列表
    sorted(a, key=lambda x: -x)
    print(a)
    a.clear()
    print(a)

def tup():
    # 元组内的元素是不能修改的
    a=1,2,3,4,5,6,7
    b=(1,2,3,4,5,6,7)
    c=(42,)
    d=()
    print(a);print(b);print(c);print(d)

    aa=tuple((1,2,3))
    bb=tuple(range(10))
    cc=tuple("sef")
    print(bb)
    # 元组是可以进行切片的

    sorted(aa, reverse=True)
    print(aa)


def zi():
    a=[1,2,3]
    b=['i','y','l']

    d=tuple(zip(a,b))   # ((10, 40), (20, 50), (30, 60))
    print(d)
    print(type(d))

    s=(x*2 for x in range(5))
    print(s)
    print(s.__next__())
    print(s.__next__())


def di():
    s={'name':'jack','age':'23','job':'pro'}
    print(s['name'])
    a = [1, 2, 3]
    b = ['i', 'y', 'l']
    b=dict(zip(a, b))


    e = [1,2,3,4,5,6]
    f = ['江','南','皮','革','厂','倒闭了']
    fv = dict(zip(f,e))

    sentence = '江 南 皮 革 厂 倒闭了'
    words = sentence.split(' ')

    print([fv[ok] for ok in words])


def setts():
    a = [1,2,3,4,4,5]
    b = set(a)
    print(b)


def joiner():
    a = 'this is my cup'
    b = a.split(' ', 2)     # 如果后边设置的有数量值，则分割后的串的数量是 n+1
    print(b)                # 切分出三个串

    c = 'to be or not cont'
    d = c.split()
    print(d)
    print('*'.join(d))

    aa = 'what\' s this'
    print(aa.upper())   # 全员大写
    print(aa.lower())   # 全员小写

    bb = 'who\'s this'
    print(bb.capitalize())  # 首字母大写
    print(bb.title())       # 每个单词的首字母大写
    print(aa.upper().swapcase())    # 大写转小写 小写转大写

    cc = 'hello Thank you'
    print(cc.startswith('H'))   # 检查首字符是不是 H
    print(cc.endswith('u'))     # 检查最后一个字符是不是 u

    ee = 'PanJiuFen'
    print(ee.center(20, '_'))   # 将原字符串放置在一个20位的字符串中间，其他位置填补 _
    print(ee.center(20))        # 将原字符串放置在一个20位的字符串中间，其他位置填补 空格

    print(ee.ljust(20, '_'))    # 原字符串在开头， 延长到20位， 其他位置补 _
    print(ee.rjust(20, '_'))    # 原字符串在结尾， 延长到20位， 其他位置补 _

    ff = '  what are you doing ?  '
    print(ff.strip())           # 删除字符串两端的空格
    print(ff.lstrip())          # 删除字符串左边的空格
    print(ff.rstrip())          # 删除字符串右边的空格

    gg = '张三遇见了卖切糕的223'
    print(gg.isalnum())         # 是否为字母或数字  如果只有字母或数字 返回True
    print(gg.isalpha())         # 是否只有字母或者汉字组成  如果只有 返回True，如果有别的 返回False
    print(gg.isdigit())         # 是否只有数字组成  如果是 返回True
    print(gg.isspace())         # 是否为空白符    如果是 返回True

    wo = 'wo'
    print('aa' + wo)

    x, y, z, = 2, 3, 4
    k = ((5 + 10 * x) / 5) - ((13 * (y - 1) * (x + y)) / x) + 9 * ((5 / x) + ((12 + x) / y))
    print(k)


def name_function(value, count):      # def 函数名 ([参数1，参数2.....])
    x = 7                           # 代码
    y = 8                           # 代码
    return x * y + value + count    # 返回


def su():
    n = 100
    value = 20
    count = 40
    return n + name_function(value, count)


def os_sys():
    d = './Adobe'
    os.listdir(d)


def open_read():
    poem = 'what are you Python 张良的风\n'
    f = open('w:/123.txt', 'w', encoding='utf-8')     # 写入内容
    f = open('w:/123.txt', 'a')     # 追加的形式加入文件
    f.write(poem)
    f.close()


def with_file():
    with open('w:/123.txt', 'r', encoding='utf-8') as f:
        data = f.read()
        print(data)


def while_file():
    f = open('w:/123.txt', 'r', encoding='utf-8')
    line = f.readline()         # 该方法 每次只读取一行内容
    while line:
        print(line)
        line = f.readline()
        # line = f.readlines()    # 该方法会读取该文件的所有行，读取大文件会很占内存
    f.close()


if __name__ == '__main__':
    open_read()
    f = open('w:/123.txt', 'r', encoding='utf-8')
    data = f.read()
    print(data)
    with_file()
    while_file()