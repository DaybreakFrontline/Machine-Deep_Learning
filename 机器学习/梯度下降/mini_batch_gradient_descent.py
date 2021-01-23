import numpy as np
# 小批量梯度下降


# 默认的从样本数中随机出来一个    这里有个问题就是有些样本可能永远也随机不到，或者经常被随机到
def default(x_count):
    return np.random.randint(x_count)


X = 2 * np.random.rand(100, 1)     # 100行1列
y = 9 + 3*X + np.random.rand(100, 1)    # 100行1列    # 真实的y

X_b = np.c_[np.ones((100, 1)), X]       # 截距项 x0    100行2列  把np.ones和X拼接到一起
print(X_b.shape)
print(X.shape[0])

# learning_rate = 0.001   # 学习率   用到动态学习率，去掉
n_epochs = 10000        # 迭代轮数
m = X.shape[0]          # 样本数100
batch_size = 10         # 每一个批次使用10条样本
num_batches = int(m / batch_size)   # 每一个轮次有多少批次

# 引用两个超参数
t0, t1 = 5, 500


# 定义一个函数来调整学习率          动态学习率    改进二
def learning_rate_schedule(t):
    return t0 / (t + t1)        # 随着迭代的次数，分母越来越大，所以学习率也会越来越小


theta = np.random.rand(2, 1)
for epoch in range(n_epochs):

    # 在双层循环之间，每个轮次开始前，打乱索引顺序    改进一
    arr = np.arange(len(X_b))
    np.random.shuffle(arr)
    X_b = X_b[arr]
    y = y[arr]

    for i in range(num_batches):
        # 随机一个索引
        # random_index = default(m)   # 这里有个问题就是有些样本可能永远也随机不到，或者经常被随机到
        # これは　一つ問題があるさ。 如果随机的小标靠末尾，那么加上的数据数会不会越界？ 不会越界，只会把剩下的元素返回
        x_batch = X_b[i*batch_size: i*batch_size + batch_size]  # 加上了个数据量   小批量梯度下降这里取部分样本
        y_batch = y[i*batch_size: i*batch_size + batch_size]    # 加上了个数据量
        # 求出梯度
        gradients = x_batch.T.dot(x_batch.dot(theta) - y_batch)
        learning_rate = learning_rate_schedule(epoch * m + i)   # 动态学习率
        print(learning_rate)
        theta = theta - learning_rate * gradients
        print(theta)

print('=======================================')
print(theta)

