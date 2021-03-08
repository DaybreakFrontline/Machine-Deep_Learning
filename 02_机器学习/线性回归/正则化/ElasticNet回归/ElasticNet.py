import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import SGDRegressor

X = 2*np.random.rand(100, 1)
y = 4 + 3*X + np.random.randn(100, 1)

elastic_reg = ElasticNet(alpha=0.02, l1_ratio=0.15)
elastic_reg.fit(X, y)
print(elastic_reg.predict([[1.5]]))
print(elastic_reg.intercept_)
print(elastic_reg.coef_)

print('============================')

sgd_reg = SGDRegressor(penalty='elasticnet', max_iter=1000)
sgd_reg.fit(X, y.ravel())
print(sgd_reg.predict([[1.5]]))
print(sgd_reg.intercept_)
print(sgd_reg.coef_)
