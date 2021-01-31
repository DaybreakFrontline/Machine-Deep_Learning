from sklearn.datasets import load_breast_cancer     # 乳腺癌数据
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

data = load_breast_cancer()
# print(data)

X, y = data['data'][:, :2], data['target']
# 求出两个维度对应的数据，在逻辑回归算法下的最优解
lr = LogisticRegression(fit_intercept=False)    # fit_intercept 是否训练截距项
lr.fit(X, y)
# 分别把两个维度所对应的参数w1和w2取出来
theta1 = lr.coef_[0, 0]       # theta是个列向量
theta2 = lr.coef_[0, 1]


# Sigmoid Function  已知w1和w2的情况下，传进来数据的x，返回数据的y_predict
def p_theta_function(features, w1, w2):
    # z = theta转置x = w1*x1 + w2*x2
    z = w1 * features[0] + w2 * features[1]
    # sigmoid函数
    return 1 / (1 + np.exp(-z))


# 逻辑回归损失函数  X, y, w1, w2。传入一份已知数据的X，y，如果已知w1和w2的情况下
# 计算对应这份数据的Loss函数
def loss_function(samples_features, samples_labels, w1, w2):
    result = 0
    # 遍历数据集中的每一条样本，并且计算每条样本的损失，加到result上得到整体的数据集损失
    for features, label in (samples_features, samples_labels):
        # 这是计算一条样本的y_predict
        p_result = p_theta_function(features, w1, w2)
        loss_result = -1 * label * np.log(p_result) - (1 - label) * np.log(1 - p_result)
        result += loss_result
    return result
