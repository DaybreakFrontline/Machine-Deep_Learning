import numpy as np
# 随机梯度下降

X = 2*np.random.rand(100, 1)            # m行1列
print(X.shape)
y = 9 + 3*X + np.random.rand(100, 1)    # m行1列
print(y.shape)
# 截距项
X_b = np.c_[np.ones((100, 1)), X]   # 100行2列

# 设置超参数
n_epochs = 1000    # 迭代轮次
m = 100     # 100条样本
# learning_rate = 0.0001   # 学习率  # 改为动态，去掉
t0, t1 = 5, 500


# 定义一个函数来调整学习率          改进二 动态学习率
def learning_rate_schedule(t):
    return t0 / (t + t1)        # 随着迭代的次数，分母越来越大，所以学习率也会越来越小


theta = np.random.rand(2, 1)    # 随机W， 一个w0 一个w1， 所以是两列
for epoch in range(n_epochs):   # 每一个轮次

    # 在双层循环之间，每个轮次开始前，打乱索引顺序  改进一
    arr = np.arange(len(X_b))
    np.random.shuffle(arr)
    X_b = X_b[arr]
    y = y[arr]

    for i in range(m):          # 每一个批次
        # 随机一个索引
        # random_index = np.random.randint(m)     # 这样的话，算是有放回的采样，因为本次随机的数下次依然会再次随机到
        xi = X_b[i: i + 1]    # 左闭右开 切割   随机梯度下降这里只取一条样本
        yi = y[i: i + 1]      # 左闭右开 切割
        # 求梯度
        gradients = xi.T.dot(xi.dot(theta) - yi)
        learning_rate = learning_rate_schedule(epoch * m + i)
        print(learning_rate)
        theta = theta - learning_rate * gradients
        print(theta)

print('=======================================')
print(theta)

