# python 的面向对象
# 当我们创建的类型不能用简单类型来表示的时候，则需要定义类
# class 首字母小写， 类名首字母大写

class Student:
    company = 'Python'
    count = 88

    def __init__(self, temp_name, temp_age):
        super().__init__()
        self.name = temp_name
        self.age = temp_age
        Student.count = Student.count + 1

    def change_age(self, new_age):
        self.age = new_age

    def get_name(self):
        return self.name

    @classmethod
    def print_information(cls):
        print(cls.company)
        print(cls.count)

    def __str__(self):
        return self.name, self.age


if __name__ == '__main__':
    student = Student('王二狗', 93)
    student.change_age(210)
    print(student.get_name())
    print(student.age)

    print(Student.company, Student.count)
    Student.print_information()

    print(Student)

