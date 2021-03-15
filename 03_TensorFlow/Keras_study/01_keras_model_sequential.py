from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation

# 使用keras来搭建神经网络    第一种

# 序列模型
# 序列模型属于通用模型的一种，因为很常见，所以这里单独列出来进行介绍
# 是依次顺序的线性关系，在第k层和第k+1层之间可以加上各种元素来构造神经网络
# 这些元素可以通过一个列表来定制，然后作为参数传递给序列模型来生成相应模型

# Dense相当于构建一个全连接层，32指的是全连接层上面神经元的个数  Dense adj. 密集的
layers = [Dense(32, input_shape=(784, )),   # 隐藏层
          Activation('relu'),               # 激活函数  relu
          Dense(10),
          Activation('softmax')]            # 激活函数  softmax
model = Sequential(layers)                  # 创建model模型
model.summary()
