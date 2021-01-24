import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor  # 梯度提升 基于决策树 用来拟合非线性数据
# %matplotlib inline    这一行是jupyter编辑器里边需要加上的，pycharm是不用加的

# 这三行是为了让控制台的输出不显示省略号
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

# 把文件数据读取进来 sep=',' 因为文件字段是用逗号分隔开的
data = pd.read_csv('./data/insurance.csv', sep=',')
print(data.head(n=6))   # 读取6行

# EDA 数据探索分析  explore data analysis
plt.hist(data['charges'])   # 取出charges这一个字段
plt.show()                  # 此时，数额大的在右边，这个现象叫做 右偏。 我们可以用np.log去处理成正太分布的图

plt.hist(np.log(data['charges']), bins=20)   # 处理成正态分布  # bins=20 让柱子更细一点
plt.show()

# 进行 特征工程
# 1、性别和是否抽烟都是只有两个状态的，而我们不能用0或1来区分这个这两种状态，而是要将这两种状态拆分成两个字段，即
#       用两个维度来表示性别或者是否抽烟。 比如性别，我们就用female和male来表示
#       而我们还有一个字段叫region,地区，那么如果这个字段有n个区域，则需要增加n个维度，不能简单地用一个字符串来定义
data = pd.get_dummies(data)
print(data.head(n=6))

# 丢掉y(实际结果)，只保留x 即影响结果的因素
x = data.drop('charges', axis=1)        # 拿到除了结果意外的因素x
y = data['charges']                     # 拿到实际结果y
x.fillna(0, inplace=True)               # 空的字段填充0
y.fillna(0, inplace=True)               # 空的字段填充0

# 区分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)   # 30%的数据作为测试集数据
# 用x_train和y_train训练模型，x_test和y_test测试模型

# 归一化   scaled adj. 等比例缩小的
scaler = StandardScaler(with_mean=True, with_std=True).fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)
print(x_train_scaled)

# 多项式回归     # 这里用2阶  要小心过拟合
poly_features = PolynomialFeatures(degree=2, include_bias=False)
# 将训练集和测试机都进行多项式回归，升2阶
x_train_scaled = poly_features.fit_transform(x_train_scaled)
x_test_scaled = poly_features.fit_transform(x_test_scaled)

# 模型训练  # 线性回归
reg = LinearRegression()    # 线性回归
reg.fit(x_train_scaled, np.log1p(y_train))    # log1p是log的优化
y_predict = reg.predict(x_test_scaled)  # 预测    # y_predict是对测试集的预测结果

ridge = Ridge()   # Ridge 岭回归的归一化
ridge.fit(x_train_scaled, np.log1p(y_train))
y_predict_ridge = ridge.predict(x_test_scaled)

booster = GradientBoostingRegressor()   # Ridge 岭回归的归一化
booster.fit(x_train_scaled, np.log1p(y_train))
y_predict_booster = booster.predict(x_test_scaled)

# 模型评估      # 把训练集的实际结果和训练集的预测结果传进来 做MSE评估  然后开根号
log_rmse_train = np.sqrt(mean_squared_error(y_true=np.log1p(y_train), y_pred=reg.predict(x_train_scaled)))
log_rmse_test = np.sqrt(mean_squared_error(y_true=np.log1p(y_test), y_pred=y_predict))
print('LinearRegression log 训练集MSE:', log_rmse_train)
print('LinearRegression log 测试集MSE:', log_rmse_test)
# 注意区分上下两种的写法
rmse_train = np.sqrt(mean_squared_error(y_true=y_train, y_pred=np.exp(reg.predict(x_train_scaled))))
rmse_test = np.sqrt(mean_squared_error(y_true=y_test, y_pred=np.exp(reg.predict(x_test_scaled))))
print('LinearRegression 训练集MSE:', rmse_train)
print('LinearRegression 测试集MSE:', rmse_test)
print('*************************************')
log_rmse_train = np.sqrt(mean_squared_error(y_true=np.log1p(y_train), y_pred=ridge.predict(x_train_scaled)))
log_rmse_test = np.sqrt(mean_squared_error(y_true=np.log1p(y_test), y_pred=y_predict_ridge))
print('ridge log 训练集MSE:', log_rmse_train)
print('ridge log 测试集MSE:', log_rmse_test)
# 注意区分上下两种的写法
rmse_train = np.sqrt(mean_squared_error(y_true=y_train, y_pred=np.exp(ridge.predict(x_train_scaled))))
rmse_test = np.sqrt(mean_squared_error(y_true=y_test, y_pred=np.exp(ridge.predict(x_test_scaled))))
print('ridge 训练集MSE:', rmse_train)
print('ridge 测试集MSE:', rmse_test)
print('*************************************')
log_rmse_train = np.sqrt(mean_squared_error(y_true=np.log1p(y_train), y_pred=booster.predict(x_train_scaled)))
log_rmse_test = np.sqrt(mean_squared_error(y_true=np.log1p(y_test), y_pred=y_predict_booster))
print('booster log 训练集MSE:', log_rmse_train)
print('booster log 测试集MSE:', log_rmse_test)
# 注意区分上下两种的写法
rmse_train = np.sqrt(mean_squared_error(y_true=y_train, y_pred=np.exp(booster.predict(x_train_scaled))))
rmse_test = np.sqrt(mean_squared_error(y_true=y_test, y_pred=np.exp(booster.predict(x_test_scaled))))
print('booster 训练集MSE:', rmse_train)
print('booster 测试集MSE:', rmse_test)


