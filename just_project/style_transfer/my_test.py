import tensorflow._api.v2.compat.v1 as tf   # 将TensorFlow2.0转换为1.0
tf.disable_v2_behavior()

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


tf.set_random_seed(42)


def test_moments():
    # 模拟创建数据X，图像维度是[batch_size, height, width]
    random_data = tf.Variable(tf.random_normal([4, 2, 3]))
    # 得到的结果是每张图片所有像素点的 均值 和 方差
    mean, var = tf.nn.moments(random_data, axes=[1, 2])

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        data, m, v = sess.run([random_data, mean, var])
        print(data)
        print(m)
        print(v)


def test_pad():
    data = [[2, 3, 4],
            [5, 6, 7]]
    paddings = [[1, 1], [2, 2]]
    mode = 'REFLECT'
    result = tf.pad(data, paddings=paddings, mode=mode)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        temp = sess.run(result)
        print(temp)


def test_transpose_conv():
    # tf.conv2d_transpose(value, filter, output_shape, strides, padding="SAME", data_format="NHWC", name=None)

    # 除去name参数用以指定该操作的name，与方法有关的一共六个参数：

    # 第一个参数value：指需要做反卷积的输入图像，它要求是一个Tensor
    # 第二个参数filter：卷积核，它要求是一个Tensor，具有[filter_height, filter_width, out_channels, in_channels]
    # 这样的shape，具体含义是[卷积核的高度，卷积核的宽度，卷积核个数，图像通道数]
    # 第三个参数output_shape：反卷积操作输出的shape，细心的同学会发现卷积操作是没有这个参数的.
    # 第四个参数strides：反卷积时在图像每一维的步长，这是一个一维的向量，长度4
    # 第五个参数padding：string类型的量，只能是"SAME", "VALID"其中之一，这个值决定了不同的卷积方式
    # 第六个参数data_format：string类型的量，'NHWC' 和 'NCHW' 其中之一，
    # 这是tensorflow新版本中新加的参数，它说明了value参数的数据格式。
    # 'NHWC' 指tensorflow标准的数据格式[batch, height, width, in_channels]
    # 'NCHW' 指Theano的数据格式, [batch, in_channels，height, width]，当然默认值是 'NHWC'

    # 通俗的讲这个解卷积，也就做反卷积，也叫做转置卷积（最贴切），我们就叫做反卷积吧，它的目的就是卷积的反向操作

    image_raw_data = tf.gfile.FastGFile('./test.jpg', mode='rb').read()
    image = tf.image.decode_jpeg(image_raw_data)

    x = tf.placeholder(tf.float32, [220, 440, 3])
    x_image = tf.reshape(x, [1, 220, 440, 3])
    real_value = tf.reshape(x_image, [220*440*3, 1])

    kernel = tf.constant(1.0, shape=[3, 3, 3, 32])
    w_kernel = tf.Variable(tf.random_normal([3, 3, 3, 32], stddev=1, seed=1))

    # y = tf.nn.conv2d(x_image, kernel, strides=[1, 2, 2, 1], padding="SAME")
    # y_image = tf.reshape(y, [110, 220])
    y = tf.nn.conv2d(x_image, w_kernel, strides=[1, 2, 2, 1], padding="SAME")

    w_kernel_2 = tf.Variable(tf.random_normal([3, 3, 3, 32], stddev=1, seed=1))
    reverse = tf.nn.conv2d_transpose(y, w_kernel_2, output_shape=[1, 220, 440, 3], strides=[1, 2, 2, 1], padding="SAME")
    reversed_image = tf.reshape(reverse, [220, 440, 3])
    predicted_value = tf.reshape(reverse, [220*440*3, 1])

    loss = tf.reduce_mean(tf.square(predicted_value - real_value))
    train = tf.train.AdamOptimizer(0.01).minimize(loss)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        result = sess.run(image)
        print(result)
        print(type(result))
        print(result.shape)

        plt.subplot(131)
        plt.imshow(result)

        # conved, reved = sess.run([y_image, reversed_image], feed_dict={x: result})
        # reved = sess.run(reversed_image, feed_dict={x: result})
        for i in range(2000):
            loss_value, reved, _ = sess.run([loss, reversed_image, train], feed_dict={x: result})
            if i % 100 == 0:
                print(i, "loss = %.4f" % loss_value)

        # print(conved)
        # print(type(conved))
        # print(conved.shape)
        #
        # plt.subplot(132)
        # plt.imshow(conved, cmap='gray')

        print(reved)
        print(type(reved))
        print(reved.shape)

        plt.subplot(133)
        plt.imshow(reved)
        plt.show()


def test_equal():
    A = [[1, 3, 4, 5, 6]]
    B = [[1, 3, 4, 3, 2]]

    with tf.Session() as sess:
        temp = sess.run(tf.equal(A, B))
        print(temp)


def test_where():
    # a,b为和tensor相同维度的tensor，将tensor中的true位置元素替换为ａ中对应位置元素，false的替换为ｂ中对应位置元素
    A = [[1, 3, 4, 5, 6]]
    B = [[1, 3, 4, 3, 2]]

    with tf.Session() as sess:
        cond = tf.equal(A, B)
        result = tf.where(cond, A, B)
        c, r = sess.run([cond, result])
        print(c)
        print(r)


def test_zeros_like():
    tensor = [[1, 2, 3],
              [4, 5, 6]]
    # 新建一个与给定的tensor类型大小一致的tensor，其所有元素为1和0
    x = tf.ones_like(tensor=tensor)
    with tf.Session() as sess:
        print(sess.run(x))


test_transpose_conv()
