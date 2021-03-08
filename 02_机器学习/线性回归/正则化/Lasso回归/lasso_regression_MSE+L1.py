import numpy as np
from sklearn.linear_model import Lasso
from sklearn.linear_model import SGDRegressor

# Lasso回归   L1范数  绝对值

X = 2*np.random.rand(100, 1)
y = 4 + 3*X + np.random.randn(100, 1)

lasso_reg = Lasso(alpha=0.01, max_iter=30000)
lasso_reg.fit(X, y)
print(lasso_reg.predict([[1.5]]))
print(lasso_reg.intercept_)
print(lasso_reg.coef_)

print('============================')

sgd_reg = SGDRegressor(penalty='l1', max_iter=10000)
sgd_reg.fit(X, np.ravel(y))
print(sgd_reg.predict([[1.5]]))
print(sgd_reg.intercept_)
print(sgd_reg.coef_)
