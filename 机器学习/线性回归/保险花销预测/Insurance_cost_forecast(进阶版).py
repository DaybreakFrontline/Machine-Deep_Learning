import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor  # 梯度提升 基于决策树 用来拟合非线性数据

# 这三行是为了让控制台的输出不显示省略号
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

data = pd.read_csv('./data/insurance.csv')
data.head()

# EDA数据探索分析
plt.hist(data['charges'])
plt.show()

plt.hist(np.log(data['charges']))
plt.show()

sns.kdeplot(data.loc[data.sex == 'male', 'charges'], shade=True, label='male')
sns.kdeplot(data.loc[data.sex == 'female', 'charges'], shade=True, label='female')
plt.show()

sns.kdeplot(data.loc[data.region == 'northwest', 'charges'], shade=True, label='northwest')
sns.kdeplot(data.loc[data.region == 'southwest', 'charges'], shade=True, label='southwest')
sns.kdeplot(data.loc[data.region == 'northeast', 'charges'], shade=True, label='northeast')
sns.kdeplot(data.loc[data.region == 'southeast', 'charges'], shade=True, label='southeast')
plt.show()

sns.kdeplot(data.loc[data.smoker == 'yes', 'charges'], shade=True, label='smoker yes')
sns.kdeplot(data.loc[data.smoker == 'no', 'charges'], shade=True, label='smoker no')
plt.show()

sns.kdeplot(data.loc[data.children == 0, 'charges'], shade=True, label='children 0')
sns.kdeplot(data.loc[data.children == 1, 'charges'], shade=True, label='children 1')
sns.kdeplot(data.loc[data.children == 2, 'charges'], shade=True, label='children 2')
sns.kdeplot(data.loc[data.children == 3, 'charges'], shade=True, label='children 3')
sns.kdeplot(data.loc[data.children == 4, 'charges'], shade=True, label='children 4')
sns.kdeplot(data.loc[data.children == 5, 'charges'], shade=True, label='children 5')
plt.show()

# 我们发现其实 性别对于最终的结果没有什么影响，因为根据性别的图像来看，性别男女的曲线基本相似，所以可以去掉
# 这些维度就叫噪声，也是特征处理的一部分，就是让对最终结果影响不大的维度去掉

# 特征工程  特征选择    # 根据刚才的分析，我们发现region和sex对最终的影响不大，所以我们把他们去掉
data = data.drop(['region', 'sex'], axis=1)
data.head()


# 我们把肥胖指数bmi来换算成离散的值，超过30的和不超过30的 # 孩子数量我们也从六个值变为两个值
def greater(df, bmi, num_child):
    df['bmi'] = 'over' if df['bmi'] >= bmi else 'un'
    df['children'] = 'no' if df['children'] == num_child else 'yes'
    return df


# greater()里边有三个参数，但是我们只给了30和0两个，这里的第一个参数就是data,data.apply实际上就是把自己当作第一个参数了
data = data.apply(greater, axis=1, args=(30, 0))
print(data.head())
# 我们可以发现，有三列是字符串类型的，把他们转换成one_hot编码
data = pd.get_dummies(data)
print(data.head())

x = data.drop('charges', axis=1)
y = data['charges']
x.fillna(0, inplace=True)
y.fillna(0, inplace=True)
print(x.head())
print(y.head())
print('--------------------')

# 切分测试集和训练集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
# 多项式升维
poly_features = PolynomialFeatures(degree=2, include_bias=False)
x_train_poly = poly_features.fit_transform(x_train)
x_test_poly = poly_features.fit_transform(x_test)

# 线性回归
reg = LinearRegression()
reg.fit(x_train_poly, np.log1p(y_train))
y_predict = reg.predict(x_test_poly)

# 评估
log_rmse_train = np.sqrt(mean_squared_error(y_true=np.log1p(y_train), y_pred=reg.predict(x_train_poly)))
log_rmse_test = np.sqrt(mean_squared_error(y_true=np.log1p(y_test), y_pred=y_predict))
print('LinearRegression log 训练集MSE:', log_rmse_train)
print('LinearRegression log 测试集MSE:', log_rmse_test)
# 注意区分上下两种的写法
rmse_train = np.sqrt(mean_squared_error(y_true=y_train, y_pred=np.exp(reg.predict(x_train_poly))))
rmse_test = np.sqrt(mean_squared_error(y_true=y_test, y_pred=np.exp(reg.predict(x_test_poly))))
print('LinearRegression 训练集MSE:', rmse_train)
print('LinearRegression 测试集MSE:', rmse_test)

ridge = Ridge()  # Ridge 岭回归的归一化
ridge.fit(x_train_poly, np.log1p(y_train))
y_predict_ridge = ridge.predict(x_test_poly)

print('*************************************')
log_rmse_train = np.sqrt(mean_squared_error(y_true=np.log1p(y_train), y_pred=ridge.predict(x_train_poly)))
log_rmse_test = np.sqrt(mean_squared_error(y_true=np.log1p(y_test), y_pred=y_predict_ridge))
print('ridge log 训练集MSE:', log_rmse_train)
print('ridge log 测试集MSE:', log_rmse_test)
# 注意区分上下两种的写法
rmse_train = np.sqrt(mean_squared_error(y_true=y_train, y_pred=np.exp(ridge.predict(x_train_poly))))
rmse_test = np.sqrt(mean_squared_error(y_true=y_test, y_pred=np.exp(ridge.predict(x_test_poly))))
print('ridge 训练集MSE:', rmse_train)
print('ridge 测试集MSE:', rmse_test)

booster = GradientBoostingRegressor()  # Ridge 岭回归的归一化
booster.fit(x_train_poly, np.log1p(y_train))
y_predict_booster = booster.predict(x_test_poly)

print('*************************************')
log_rmse_train = np.sqrt(mean_squared_error(y_true=np.log1p(y_train), y_pred=booster.predict(x_train_poly)))
log_rmse_test = np.sqrt(mean_squared_error(y_true=np.log1p(y_test), y_pred=y_predict_booster))
print('booster log 训练集MSE:', log_rmse_train)
print('booster log 测试集MSE:', log_rmse_test)
# 注意区分上下两种的写法
rmse_train = np.sqrt(mean_squared_error(y_true=y_train, y_pred=np.exp(booster.predict(x_train_poly))))
rmse_test = np.sqrt(mean_squared_error(y_true=y_test, y_pred=np.exp(booster.predict(x_test_poly))))
print('booster 训练集MSE:', rmse_train)
print('booster 测试集MSE:', rmse_test)
