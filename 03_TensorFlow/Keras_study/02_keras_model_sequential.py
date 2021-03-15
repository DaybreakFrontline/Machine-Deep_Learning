from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation

# 使用keras来搭建神经网络    第二种

# 首先创建一个model模型，里边是空的
model = Sequential()
# 设置输入和隐藏节点
model.add(Dense(32, input_shape=(784, )))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))
model.summary()