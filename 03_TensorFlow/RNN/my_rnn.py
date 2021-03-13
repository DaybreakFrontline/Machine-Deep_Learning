import tensorflow._api.v2.compat.v1 as tf
import matplotlib.pyplot as plt
import numpy as np
tf.compat.v1.disable_eager_execution()

# 使用RNN循环神经网络实现手写数字数据集  10分类

# 隐藏层节点数
n_hidden_units = 128
epochs = 10000
batch_size = 1000
n_classes = 10
lr = 0.01   # 学习率

# 使用四个集合去接应手写数字数据集
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# (60000, 28, 28) (60000,) (10000, 28, 28) (60000,)
print(x_train.shape, y_train.shape, x_test.shape, y_train.shape)

# 打印出第一张图片  图片区间就是0-255之间， 0就是黑色 255就是白色
plt.imshow(x_train[0], cmap="gray")
# plt.show()

# 归一化   把所有数归到 0-1之间
x_train, x_test = x_train / 255.0, x_test / 255.0

# 转换为tf的float32类型， 接受的数组维度必须是三维的，且第一个维度任意，第二个第三个必须是28
X_ = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28])
y_ = tf.placeholder(dtype=tf.int32, shape=[None])

# 把RNN构建出来
# num_units隐藏层节点数量 该参数为必要参数
cell = tf.nn.rnn_cell.BasicRNNCell(num_units=n_hidden_units)
# 把初始的隐藏状态设置为0, 创建多少个0
init_state = cell.zero_state(batch_size=batch_size, dtype=tf.float32)
# 横向的穿成一个串  tf.nn.bidirectional_dynamic_rnn:双向的rnn.  dynamic_rnn：单项的rnn
# 建立RNN  cell隐藏层节点数     input输入的数据   init_state初始化数据    time_major每一个时刻传入的
outputs, state = tf.nn.dynamic_rnn(cell=cell, inputs=X_, initial_state=init_state, time_major=False)

# 算出向上的输出
weights = tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
# 截距项
biases = tf.Variable(tf.constant(0.1, shape=[n_classes, ]))

# matmul点乘  z：经过softmax前面的z     batch_size行  10列
z = tf.matmul(state, weights) + biases

y = tf.one_hot(y_, depth=10, dtype=tf.float32)
# sparse 稀疏的    reduce_mean:求平均
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=z, labels=y_))
# 最小化loss
train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
# argmax 最大的    对比z和y是否相同
correct_pred = tf.equal(tf.argmax(z, 1), tf.argmax(y, 1))
# reduce_mean平均值    cast：把correct_pred转成float32格式
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 初始化变量
init = tf.global_variables_initializer()

# 训练集数量除batch_size 切片
num_batches = int(np.ceil(x_train.shape[0] / batch_size))

with tf.Session() as sess:
    sess.run(init)

    # 创建每个批次的数据
    for epochs in range(epochs):
        arr = np.arange(x_train.shape[0])
        np.random.shuffle(arr)
        x_train = x_train[arr]
        y_train = y_train[arr]

        # 每隔一轮，计算一次准确率 accuracy 准确率
        acc = sess.run(accuracy, feed_dict={
            X_: x_test,
            y_: y_test
        })
        print("Test set accuracy %s" % acc)

        for batch in range(num_batches):
            x_batches = x_train[batch * batch_size: batch * batch_size + batch_size]
            y_batches = y_train[batch * batch_size: batch * batch_size + batch_size]

            sess.run(train_op, feed_dict={
                X_: x_batches,
                y_: y_batches
            })