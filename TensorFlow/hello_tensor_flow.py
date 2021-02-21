import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
first_image = x_train[0]
first_label = y_train[0]
print(first_image, first_label)

plt.imshow(first_image)
plt.show()

# 归一化 归到0-1之间
x_train, x_test = x_train / 255.0, x_test / 255.0

# Sequential 序列化，
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),    # 输入形状，注意这里边的是一个样本的参数   Flatten:压平， 28*28 => 784*1
  tf.keras.layers.Dense(128, activation='relu'),    # 128个隐藏神经元， 激活函数为relu
  tf.keras.layers.Dropout(0.2),                     # Dropout 丢弃， 将20%神经元数据丢弃，只传递80%。 目的是防止过拟合
  tf.keras.layers.Dense(10, activation='softmax')   # 10个输出节点， softmax分类任务， 10分类。 Dense 稠密的。 sparse 稀疏的。
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)
