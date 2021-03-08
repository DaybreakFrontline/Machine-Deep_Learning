# 在import tensorflow之前加上 会强制使用CPU运行
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.client import device_lib
# 查看是用的CPU还是GPU运行的
print(device_lib.list_local_devices())

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# first_image = x_train[0]
# first_label = y_train[0]
# print(first_image, first_label)

# plt.imshow(first_image)
# plt.show()

# 归一化 归到0-1之间
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train, x_test = np.reshape(x_train, (-1, 28, 28, 1)), np.reshape(x_test, (-1, 28, 28, 1))

# (x_train, x_test) = np.reshape(x_train, (60000, 784)), np.reshape(x_test, (10000, 784))

# Sequential 序列化，
model = tf.keras.models.Sequential([  # 返回了一个网络拓扑模型
    tf.keras.layers.Input(shape=(28, 28, 1)),  # 这一行是输入    改成 shape=(784,) 也行。 那么就要把x_train和x_test reshape成一维的
    # tf.keras.layers.Flatten(),  # 输入形状，注意这里边的是一个样本的参数   Flatten:压平， 28*28 => 784*1
    # tf.keras.layers.Flatten(shape=(28, 28)),        # 上边两行可以合并成这一行

    # 加入卷积层
    tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3)),
    tf.keras.layers.MaxPool2D(),

    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3)),
    tf.keras.layers.MaxPool2D(),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation='relu'),  # 128个隐藏神经元， 激活函数为relu
    tf.keras.layers.Dropout(0.2),  # Dropout 丢弃， 将20%神经元数据丢弃，只传递80%。 目的是防止过拟合
    tf.keras.layers.Dense(10, activation='softmax')  # 10个输出节点， softmax分类任务， 10分类。 Dense 稠密的。 sparse 稀疏的。
])

model.compile(optimizer='adam',  # optimizer 优化器。 梯度下降也是一种optimizer。 用adam的方式去优化参数
              loss='sparse_categorical_crossentropy',  # 损失函数 loss: 稀疏的_分类_交叉熵。 把y_test做成 oneHot编码，所以变得稀疏了。
              # 这里如果不用 sparse，那就提前对y_train和y_test提前做 oneHot编码处理
              metrics=['accuracy'])  # 评估指标  accuracy 对的样本数/总样本数

model.fit(x_train, y_train, epochs=10)  # 训练
model.evaluate(x_test, y_test)  # 测试    碰到 Dropout 时，会直接跳过 Dropout
