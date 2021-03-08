# 子类和父类
# Py是可以多继承的  class origin(apple, banana)

class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def sing(self):
        print("i want sing")


class Student(Person):
    def __init__(self, score, name, age):
        self.score = score
        # 子类并不会自动调用父类的init构造方法，需要显性的调用
        Person.__init__(self, name, age)

    def dance(self):
        print("i want dance")

    def sing(self):
        # 调用父类的 sing 方法
        super().sing()


class Teacher(Student):
    pass


def __test():
    print('private 的 function')


if __name__ == '__main__':
    s1 = Student(94, '张三', 22)
    s1.dance()
    s1.sing()
    print(s1.score, s1.age, s1.name)

    t1 = Teacher(94, '张三', 22)
    t1.dance()
    t1.sing()
