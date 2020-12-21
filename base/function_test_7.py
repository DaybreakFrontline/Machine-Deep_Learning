# python 的面向对象
# 当我们创建的类型不能用简单类型来表示的时候，则需要定义类
# class 首字母小写， 类名首字母大写

class Student:
    def __init__(self, name, score):
        super().__init__()
        self.name = name
        self.score = score

    def say_score(self):
        print(self.name, '的分数是:', self.score)


s1 = Student('王二狗', 93)
s1.say_score()
