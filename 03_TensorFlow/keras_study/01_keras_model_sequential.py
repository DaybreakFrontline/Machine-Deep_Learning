from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.datasets.mnist import m

# Dense相当于构建一个全连接层，32指的是全连接层上面神经元的个数
layers = [Dense(32, input_shape=(784, ))]
