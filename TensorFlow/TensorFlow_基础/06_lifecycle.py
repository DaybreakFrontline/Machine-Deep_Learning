# 生命周期  48:14
import tensorflow as tf

# 当去计算一个节点的时候，TensorFlow自动计算他一来的一组节点，并且首先计算依赖的节点
w = tf.Variable(3)
x = w + 2
y = x + 5
z = x * 3


with tf.compat.v1.Session() as sess:
    sess.run(w.initializer)
    print(sess.run(y))
    # 这里为了计算z, 有重新计算了x和w, 除了Variable值，tf是不会缓存其他比如contant等的值
    # 一个Variable的生命周期是当它的Initializer运行的时候开始，到回话session close的时候结束
    print(sess.run(z))

# 如果我们想要有效的计算y和z，并且有不重复计算w和x两次，我们必须要求TensorFlow计算y和z在一个图中
with tf.compat.v1.Session() as sess:
    sess.run(w.initializer)
    y_val, z_val = sess.run([y, z])
    print(y_val)
    print(z_val)