import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 多项式回归  升维度

np.random.seed(42)
m = 130
X = 6*np.random.rand(m, 1) - 3
y = 0.5*X**2 + X + 2 + np.random.randn(m, 1)    # 0.5*X^2:w2 / X:w1 / 2:w0
plt.plot(X, y, 'b.')    # b. blue color
# plt.show()

X_train = X[:80]
y_train = y[:80]
X_test = X[80:]
y_test = y[80:]

d = {1: 'g-', 2: 'r+', 10: 'y*'}    # 根据图，1维欠拟合， 10维过拟合
for i in d:
    poly_features = PolynomialFeatures(degree=i, include_bias=True)
    X_poly_train = poly_features.fit_transform(X_train)
    X_poly_test = poly_features.fit_transform(X_test)
    print('---------------')
    print(X_train[0])
    print(X_poly_train[0])
    print(X_train.shape)
    print(X_poly_train.shape)
    print('===============')
    lin_reg = LinearRegression(fit_intercept=False)
    lin_reg.fit(X_poly_train, y_train)
    print(lin_reg.intercept_, lin_reg.coef_)

    # 看看是否随着degree增加升维，是否过拟合
    y_train_predict = lin_reg.predict(X_poly_train)
    y_test_predict = lin_reg.predict(X_poly_test)

    plt.plot(X_poly_train[:, 1], y_train_predict, d[i])    # 1代表第二列
    print('++++++++++++++')
    # 对比三次的打印输出，就可以发现维度等于10的时候，误差不降反增，这就是过拟合了
    print(mean_squared_error(y_train, y_train_predict))
    print(mean_squared_error(y_test, y_test_predict))

plt.show()