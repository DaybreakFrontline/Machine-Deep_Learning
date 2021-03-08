import tensorflow as tf

# 这个是TensorFlow1.0的写法
# 1、构建graph图
# 2、启动session，运行graph图
# TensorFlow1.0构建的graph图是静态的，而在TensorFlow2.0是动态的 DynamicGraph

x = tf.Variable(3, name='x')
print(x)
y = tf.Variable(4, name='y')
print(y)

f = x*x*y + y + 2

# 我们不必对每个变量都单独初始化，如果我们之前有一万个变量，那么岂不是要初始化一万次
# 我们不立刻去初始化，等run的时候再去
init = tf.global_variables_initializer()


with tf.compat.v1.Session() as sess:
    sess.run(init)
    result = f.eval()

print(result)