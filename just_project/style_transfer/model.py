import tensorflow._api.v2.compat.v1 as tf   # 将TensorFlow2.0转换为1.0
tf.disable_v2_behavior()
import numpy as np

# 损失网络使用TensorFlow Slim的VGG16模型，它的实际定义位置是在nets/vgg.py文件中


def relu(input):
    relu = tf.nn.relu(input)
    # 当activation function为relu的时候，有可能会导致输出比较大，这样在取e的x方的时候，会把整个数字弄的特别大，然后会出nan
    # 相对来说，tanh, sigmoid的值域是在[-1, 1] / [0, 1]的范围之内。这两个函数不会出现nan的情况
    # 但是相对于sigmoid, sgd函数来说，relu函数训练速度比较快
    # 把空 NAN 变为 0
    # convert nan to zero (nan != nan)
    nan_to_zero = tf.where(tf.equal(relu, relu), relu, tf.zeros_like(relu))
    return nan_to_zero


def conv2d(x, input_filters, output_filters, kernel, strides, mode='REFLECT'):
    with tf.variable_scope('conv'):
        shape = [kernel, kernel, input_filters, output_filters]
        weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weight')
        # 因为图像是[batchsize, height, width, kernels]四个维度的，所以第二个参数是列表中四个元素
        # 下面说明只对height，width的上下添加行或列，REFLECT模式说明像照镜子似的对称填充
        x_padded = tf.pad(x, [[0, 0], [np.int(kernel / 2), np.int(kernel / 2)], [np.int(kernel / 2), np.int(kernel / 2)], [0, 0]], mode=mode)
        return tf.nn.conv2d(x_padded, weight, strides=[1, strides, strides, 1], padding='VALID', name='conv')


def instance_norm(x):
    """
    所谓的instance_norm说白了就是只有一个实例的batch_norm
    :param x: 可以理解为我们输出的数据，形如 [batchsize, height, width, kernels]
    :return:
    """
    epsilon = 1e-9
    # moments就是对x数据按照[1,2]这两个维度来求均值方差，正常3个维度就是[batch, height, width]
    # 得到的结果是每张图片所有像素点的 均值 和 方差
    mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
    return tf.div(tf.subtract(x, mean), tf.sqrt(tf.add(var, epsilon)))


def residual(x, filters, kernel, strides):
    with tf.variable_scope('residual'):
        conv1 = conv2d(x, filters, filters, kernel, strides)
        conv2 = conv2d(relu(conv1), filters, filters, kernel, strides)
        result = conv2 + x
        return result


def conv2d_transpose(x, input_filters, output_filters, kernel, strides):
    with tf.variable_scope('conv_transpose'):
        # input_filters: 是反卷积之前的数据的channel数量
        # output_filters: 是反卷积之后要恢复出来的channel数量
        shape = [kernel, kernel, output_filters, input_filters]
        weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weight')

        batch_size = tf.shape(x)[0]
        height = tf.shape(x)[1] * strides
        width = tf.shape(x)[2] * strides
        output_shape = tf.stack([batch_size, height, width, output_filters])
        return tf.nn.conv2d_transpose(x, weight, output_shape, strides=[1, strides, strides, 1], name='conv_transpose')


def resize_conv2d(x, input_filters, output_filters, kernel, strides, training):
    # 另外一种反卷积的方式
    # See http://distill.pub/2016/deconv-checkerboard
    with tf.variable_scope('conv_transpose'):
        height = x.get_shape()[1].value if training else tf.shape(x)[1]
        width = x.get_shape()[2].value if training else tf.shape(x)[2]

        new_height = height * strides * 2
        new_width = width * strides * 2
        x_resized = tf.image.resize_images(x, [new_height, new_width], tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        # shape = [kernel, kernel, input_filters, output_filters]
        # weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weight')
        return conv2d(x_resized, input_filters, output_filters, kernel, strides)


# 定义图像生成网络
def net(image, training):
    # 一开始在图片的上下左右加上一些额外的"边框"，目的是消除边缘效应
    image = tf.pad(image, [[0, 0], [10, 10], [10, 10], [0, 0]], mode='REFLECT')

    # 三层卷积层
    with tf.variable_scope('conv1'):
        conv1 = relu(instance_norm(conv2d(image, 3, 32, 9, 1)))
    with tf.variable_scope('conv2'):
        conv2 = relu(instance_norm(conv2d(conv1, 32, 64, 3, 2)))
    with tf.variable_scope('conv3'):
        conv3 = relu(instance_norm(conv2d(conv2, 64, 128, 3, 2)))

    # 仿照ResNet定义一些跳过连接
    with tf.variable_scope('res1'):
        res1 = residual(conv3, 128, 3, 1)
    with tf.variable_scope('res2'):
        res2 = residual(res1, 128, 3, 1)
    with tf.variable_scope('res3'):
        res3 = residual(res2, 128, 3, 1)
    with tf.variable_scope('res4'):
        res4 = residual(res3, 128, 3, 1)
    with tf.variable_scope('res5'):
        res5 = residual(res4, 128, 3, 1)

    # 定义卷积之后定义反卷积
    # 反卷积不采用通常的转置卷积的方式，而是采用先放大，再做卷积的方式
    # 这样可以消除噪声点
    with tf.variable_scope('deconv1'):
        deconv1 = relu(instance_norm(resize_conv2d(res5, 128, 64, 3, 2, training)))

    with tf.variable_scope('deconv2'):
        deconv2 = relu(instance_norm(resize_conv2d(deconv1, 64, 32, 3, 2, training)))

    with tf.variable_scope('deconv3'):
        deconv3 = tf.nn.tanh(instance_norm(conv2d(deconv2, 32, 3, 9, 1)))

    # tanh的输出范围是-1到1的，知道RGB图像的像素范围是0到255的，所以这里对deconv3进行缩放
    y = (deconv3 + 1) * 127.5

    # 最后去除一开始为了防止边缘效应而加入的边框
    height = tf.shape(y)[1]
    width = tf.shape(y)[2]
    y = tf.slice(y, [0, 10, 10, 0], tf.stack([-1, height-20, width-20, -1]))

    return y



