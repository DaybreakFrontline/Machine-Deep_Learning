

def hundrond():
    data = list(range(1, 101))
    return data


def add(a, b):
    if len(a) == 2:
        return a[0] + a[1] + b
    return add(a[:-1], a[-1]) + b


def sum():
    a = 0
    for i in range(1, 11):
        a += 1
    print(a)


if __name__ == '__main__':
    data = list(range(1, 101))

    print(data[:-1])
    print(data[-1])

    value = add(data[:-1], data[-1])
    print(value)
    sum()