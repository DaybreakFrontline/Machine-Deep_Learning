import numpy as np
# 全量梯度下降

# 创建数据集 X, y    有监督机器学习
X = np.random.rand(100, 1)
y = 9 + 3 * X + np.random.randn(100, 1)
X_b = np.c_[np.ones((100, 1)), X]   # 添加x0

# 创建超参数
# learning_rate = 0.one    # 学习率   # 改为动态，去掉
n_iterations = 10000    # 迭代次数

# 引用两个超参数
t0, t1 = 5, 500


# 定义一个函数来调整学习率          改进二 动态学习率
def learning_rate_schedule(t):
    return t0 / (t + t1)        # 随着迭代的次数，分母越来越大，所以学习率也会越来越小


# 1、初始化 theta, W0 ... Wn, 标准正态分布创建W
theta = np.random.randn(2, 1)

# 4、去判断是否收敛。 一般不会去设定阈值，而是直接采用设置相对大的迭代次数来保证收敛
for i in range(n_iterations):
    # 2、求梯度     .T 转置  X_b.T = 2行m列，  (X_b.dot(theta) - y) = epsilon = m行1列
    gradients = X_b.T.dot(X_b.dot(theta) - y)    # 等于2行1列的列向量

    # 3、应用梯度下降公式，去调整theta值  theta t+1 = theta t - η(learning_rate) * gradients(梯度)
    learning_rate = learning_rate_schedule(i)   # 动态学习率
    theta = theta - learning_rate * gradients
    print(learning_rate)
    print(theta)

print('=======================================')
print(theta)