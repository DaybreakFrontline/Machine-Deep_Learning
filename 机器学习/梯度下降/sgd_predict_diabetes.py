import numpy as np
from sklearn.datasets import load_diabetes
# 随机小批量梯度下降

data = load_diabetes()              # diabetes 糖尿病数据集 442个实数
feature_names = data.feature_names
X, y = data.data, data.target
num_examples = X.shape[0]           # X:ndarray, 拿到多少行
X = np.concatenate((np.ones((num_examples, 1)), X), axis=1)   # m行n+1列
print(X.shape, y.shape)
y = np.reshape(y, (-1, 1))          # 变形，行向量变列向量
print(y.shape)

learning_rate = 0.001   # 学习率
batch_size = 200        # 批次
num_epochs = 1000       # 轮次
num_batches = int(np.ceil(num_examples / batch_size))   # 每一轮的批次，向上取整

W = np.random.randn(X.shape[1])     # 第一步，随机一个 n+1行1列的列向量
print(W.shape)
W = np.reshape(W, (-1, 1))
print(W.shape)

previous_loss = 100000
count = 0

for epoch in range(num_epochs):

    # 打乱我们的数据
    indexes = np.asarray(range(num_examples))
    np.random.shuffle(indexes)
    X, y = X[indexes], y[indexes]

    for batch in range(num_batches):
        X_batches = X[batch*batch_size: (batch+1) * batch_size]
        y_batches = y[batch*batch_size: (batch+1) * batch_size]

        error = (np.dot(X_batches, W) - y_batches)
        grad = (2/X_batches.shape[0]) * np.dot(np.transpose(X_batches), error)    # 求梯度核心公式
        print(error.shape, grad.shape)

        W = W - learning_rate * grad
        # W = W - learning_rate * (grad + W)
        print(W.shape)

        current_loss = MSE = np.mean(np.square(error))

        detla_loss = previous_loss - current_loss
        if 0 < detla_loss < 0.001:
            count += 1
        else:
            count = 0

        if count == 10:
            break

        previous_loss = current_loss
        print(epoch, batch, current_loss)

print(W)


