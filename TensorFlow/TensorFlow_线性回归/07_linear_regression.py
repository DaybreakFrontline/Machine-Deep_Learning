import numpy as np
from sklearn.datasets import fetch_california_housing
import tensorflow._api.v2.compat.v1 as tf   # 将TensorFlow2.0转换为1.0
tf.disable_v2_behavior()

# 立刻下载数据集
housing = fetch_california_housing(data_home="Q:/尚学堂人工智能/百战人工智能33期1组/10_深度学习入门2：工欲善其事必先利其器，"
                                             "TensorFlow使用，剖析MNIST手写数字识别代码/california_housing",
                                   download_if_missing=True)
# 获得X数据行数和列数
m, n = housing.data.shape
print(m, n)
print(housing.data, housing.target)
print(housing.feature_names)

# 这里添加一个额外的bias输入特征(x0 = 1)到所有的训练数据上面，因为使用的numpy所以会立即执行
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]   # 设置一个截距项
# 创建两个TensorFlow常量节点X和y， 去持有数据和标签
X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name='X')
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name='y')     # reshape(-1, 1) 把横向量转换为列向量
# 使用一些TensorFlow框架提供的矩阵操作去求 θ theta
XT = tf.transpose(X)
# 解析解一步计算出最优解   tf.matmul(XT, X):XT和X矩阵相乘。 tf.matrix_inverse 逆矩阵。
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)

print(theta)

