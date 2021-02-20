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
print(f)

# 如果是 TensorFlow2.0的话，下边的代码就不用启动了。因为计算式动态的

# 创建一个计算图的上下文环境
# 配置里面是把具体运行过程在哪里执行给打印出来    log_device_placement=True 会打印设备日志

# with块
with tf.compat.v1.Session as sess:
    sess.run(x.initializer)
    sess.run(y.initializer)
    result = f.eval()

print(result)