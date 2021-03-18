# NLP感情分析处理
# 通过电影评论的例子详细的讲解DL在感情分析中的关键

import numpy as np
from tensorflow.keras.datasets import imdb
import matplotlib.pyplot as plt

# Keras自带的load_data函数下载了亚马逊S3上的数据，并且给每个词一个索引（index）
# 创建了字典。每段文字的每一个词对应了一个数字
(X_train, y_train), (X_test, y_test) = imdb.load_data()
print(X_train[0])
print((len(X_train[0])))
print(X_train[1])
print(len(X_train[1]))

# 1表示正面，0表示负面(25000,)
print(X_train.shape)
print(y_train.shape)
print(y_train[:10])

# 打印平均每个评论有多少个字
avg_len = list(map(len, X_train))
print(max(avg_len))
print(np.mean(avg_len))
# 为了直观显示，这里画一个分布图
plt.hist(avg_len, bins=range(min(avg_len), max(avg_len) + 100, 10))
plt.show()