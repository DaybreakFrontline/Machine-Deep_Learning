import tensorflow._api.v2.compat.v1 as tf   # 将TensorFlow2.0转换为1.0
tf.disable_v2_behavior()

from StyleTransfer.style_transfer import reader
from StyleTransfer.style_transfer import model
from StyleTransfer.style_transfer.train import get_preprocessing

import time
import os

tf.flags.FLAGS.DEFINE_string('loss_model', 'vgg_16', 'The name of architecture to evaluate.')
tf.flags.FLAGS.DEFINE_integer('image_size', 256, 'Image size to train.')
tf.flags.FLAGS.DEFINE_string('model_file', 'fast-style-model.ckpt-done', '')
tf.flags.FLAGS.DEFINE_string('image_file', 'test.jpg', '')

FLAGS = tf.flags.FLAGS


def main(_):

    height = 0
    width = 0
    with open(FLAGS.image_file, 'rb') as img:
        with tf.Session().as_default() as sess:
            if FLAGS.image_file.lower().endswith('png'):
                image = sess.run(tf.image.decode_png(img.read()))
            else:
                image = sess.run(tf.image.decode_jpeg(img.read()))
            height = image.shape[0]
            width = image.shape[1]
    tf.logging.info('Image size: %dx%d' % (width, height))

    with tf.Graph().as_default():
        with tf.Session().as_default() as sess:

            # 读取图片数据
            image_preprocessing_fn, _ = get_preprocessing('vgg16')
            image = reader.get_image(FLAGS.image_file, height, width, image_preprocessing_fn)

            # 添加批次维度
            image = tf.expand_dims(image, 0)

            generated = model.net(image, training=False)
            generated = tf.cast(generated, tf.uint8)

            # 删除批次维度
            generated = tf.squeeze(generated, [0])

            # 复原模型变量
            saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V1)
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

            # 使用绝对路径
            FLAGS.model_file = os.path.abspath(FLAGS.model_file)
            saver.restore(sess, FLAGS.model_file)

            # 保证'generated'文件夹存在
            generated_file = 'generated/res.jpg'
            if os.path.exists('generated') is False:
                os.makedirs('generated')

            # 产生和写入图片数据到文件
            with open(generated_file, 'wb') as img:
                start_time = time.time()
                img.write(sess.run(tf.image.encode_jpeg(generated)))
                end_time = time.time()
                tf.logging.info('Elapsed time: %fs' % (end_time - start_time))

                tf.logging.info('Done. Please check %s.' % generated_file)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()

















