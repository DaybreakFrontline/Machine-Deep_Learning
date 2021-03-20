# 循环神经网络训练感情分析
# 使用长短时记忆模型LSTM处理感情分类
# LSTM是循环神经网络的一种，本质上是按照时间顺序，把信息有效的真核
# LSTM根据受伤的训练数据，找到一个方法来有效的做信息取舍

from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Embedding           # 嵌入层
from tensorflow.keras.preprocessing import sequence     # keras里边的预处理，截断
import numpy as np
from tensorflow.keras.datasets import imdb

(X_train, y_train), (X_test, y_test) = imdb.load_data()
max_word = 400
X_train = sequence.pad_sequences(X_train, maxlen=max_word)
X_test = sequence.pad_sequences(X_test, maxlen=max_word)
vocab_size = np.max([np.max(X_train[i]) for i in range(X_train.shape[0])]) + 1
print("vocab_size:", vocab_size)

# no Flatten    时刻没有定义
model = Sequential()
model.add(Embedding(vocab_size, 64, input_length=max_word))
model.add(LSTM(128, return_sequences=True))     # return_sequences 往上传三维数组
model.add(Dropout(0.2))
model.add(LSTM(64, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(32, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(16))                             # 往上传二维数组
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))

'''
model = Sequential()
model.add(Embedding(vocab_size, 64, input_length=max_word))
model.add(LSTM(128, return_sequences=True))     # return_sequences 往上传三维数组
model.add(Dropout(0.2))
model.add(LSTM(64, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(32)
model.add(Dropout(0.2))                         # 往上传二维数组
model.add(Dense(1, activation='sigmoid'))
'''

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
print(model.summary())

# 导入数据，训练模型     validation_data 验证数据
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=50, verbose=1)
# evaluate .v 	评估、估计
score = model.evaluate(X_test, y_test)
print(score)
# 其精度大约在86.7%，如果迭代更多次，精度会更高

