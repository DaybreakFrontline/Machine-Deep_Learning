# 多层全连接神经网络训练感情分析
# Keras提供了设计嵌入层的模板，只要在建模的时候加入一行Embedding Layer的函数

from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Embedding           # 嵌入层
from tensorflow.keras.preprocessing import sequence     # keras里边的预处理，截断
import numpy as np
from tensorflow.keras.datasets import imdb
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"     # 使用第一, 二块GPU

(X_train, y_train), (X_test, y_test) = imdb.load_data()
# 使用下面的命令计算最长的文本长度
m = max(max(list(map(len, X_train))), max(list(map(len, X_test))))
print(m)

# 从中我们会发现有一个文本特别的长，有2494个字符，这种异常值需要排除，
# 考虑到全部文本的平均长度为238个字符，我们设定最长输入的文本为400个字符，不足400个字符的使用空格填充
# 超过400个字符的文本截取到400个字符
max_word = 400

# pad_sequences填充，maxlen=最长是多长，把训练和测试都截取或填充到400个字符的长度
X_train = sequence.pad_sequences(X_train, maxlen=max_word)
X_test = sequence.pad_sequences(X_test, maxlen=max_word)
print(X_train.shape)    # (25000, 400)
vocab_size = np.max([np.max(X_train[i]) for i in range(X_train.shape[0])]) + 1
print("vocab_size:", vocab_size)
# 这里的1代表空格，其索引是0

# 下面从最简单的多层神经网络开始尝试
# 首先建立序列模型，逐步网上搭建网络
model = Sequential()

# 第一层是嵌入层，定义了嵌入层的矩阵为vocab_size*64，每个训练段落为其中的max_word*64    # Embedding  嵌入式 词嵌入层
# vocab_size行，64列
model.add(Embedding(vocab_size, 64, input_length=max_word))     # vocab_size 字典长度

# 矩阵，作为数据的输入，填入输出层
# 把输入层压平，原来是max_word*64的矩阵，现在变成一维的长度为max_word*64的向量，
model.add(Flatten())
# 接下来不断搭建全连接神经网络，使用relu函数，最后一层是Sigmoid，预测0,1
model.add(Dense(500, activation='relu'))    # Dense adj. 稠密的 全连接层
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# epochs=20, batch_size=50  accuracy: 0.8577

# compile v. 编译     optimizer n. 优化器     metrics n. 衡量指标     accuracy n. 准确(性)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# summary n. 概括、概要
print(model.summary())

# 导入数据，训练模型     validation_data 验证数据
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=50, verbose=1)
# evaluate .v 	评估、估计
score = model.evaluate(X_test, y_test)
print(score)
# 其精度大约在85%，如果迭代更多次，精度会更高