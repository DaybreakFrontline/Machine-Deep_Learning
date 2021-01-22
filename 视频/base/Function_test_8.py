# 运算符重载

class Student(object):
    company = '学习'
    count = 0

    def __init__(self, temp_name, temp_age):
        self.name = temp_name
        self.age = temp_age

    def __add__(self, other):
        return self.age + other.age


if __name__ == '__main__':
    a = 20
    b = 30
    print(a + b)
    print(a.__add__(b))

    s1 = Student('张三', 19)
    s2 = Student('李四', 22)
    print(s1 + s2)  # +等同于__add__方法
