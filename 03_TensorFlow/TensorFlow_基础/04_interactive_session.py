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

init = tf.global_variables_initializer()

# 另外一种创建session的方法  （03_TensorFlow v1.0）
sess = tf.InteractiveSession()
init.run()
result = f.eval()
print(result)
sess.close()