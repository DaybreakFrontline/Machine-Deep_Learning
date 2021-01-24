import numpy as np
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDRegressor

# Ridge岭回归  L2范数  平方    常用
# Σ(xw - y)^2 + αΣ+（w）^2

# rand:随机样本   randn:标准正态分布中返回
X = 2*np.random.rand(100, 1)
y = 4 + 3*X + np.random.randn(100, 1)

# alpha:正则项系数   solver:sag:随机梯度下降
ridge_reg = Ridge(alpha=0.3, solver='sag')
ridge_reg.fit(X, y)
print(ridge_reg.predict([[1.5]]))   # 预测一个值，输入二维
print(ridge_reg.intercept_)         # 截距项
print(ridge_reg.coef_)              # 其他系数

print('=========================================')

sgd_reg = SGDRegressor(penalty='l2', max_iter=1000)
# 这样写会报警告，只要我们换成行向量即可
# sgd_reg.fit(X, y)
# 变成行向量
sgd_reg.fit(X, y.reshape(-1))
print(sgd_reg.predict([[1.5]]))
print(sgd_reg.intercept_)
print(sgd_reg.coef_)