import tensorflow._api.v2.compat.v1 as tf   # 将TensorFlow2.0转换为1.0
tf.disable_v2_behavior()
import tensorflow as tf2

from just_project.style_transfer import reader
from just_project.style_transfer import model
from just_project.style_transfer.train import get_preprocessing

import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"     # 使用第一, 二块GPU

tf2.compat.v1.flags.DEFINE_string('loss_model', 'vgg_16', 'The name of architecture to evaluate.')
tf2.compat.v1.flags.DEFINE_integer('image_size', 512, 'Image size to train.')
tf2.compat.v1.flags.DEFINE_string('model_file', 'models/wave/xingkong_fast-style-model.ckpt-14000', '')


name_list = []
read_directory = "source/"
for filename in os.listdir(read_directory):
    if filename.split('.')[1] == 'jpg':
        print(filename)  # 仅仅是为了测试
        name_list.append(filename)


def main(_):

    for i in range(len(name_list)):
        jpg_name = read_directory + name_list[i]
        # tf2.compat.v1.flags.DEFINE_string('image_file', jpg_name, '')
        FLAGS = tf.flags.FLAGS

        try:
            height = 0
            width = 0
            with open(jpg_name, 'rb') as img:
                with tf.Session().as_default() as sess:
                    if jpg_name.lower().endswith('png'):
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
                    image = reader.get_image(jpg_name, height, width, image_preprocessing_fn)

                    # 添加批次维度
                    image = tf.expand_dims(image, 0)

                    generated = model.net(image, training=False)
                    generated = tf.cast(generated, tf.uint8)

                    # 删除批次维度
                    generated = tf.squeeze(generated, [0])

                    # 复原模型变量
                    saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V1)
                    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

                    # 使用绝对路径  模型文件
                    FLAGS.model_file = os.path.abspath(FLAGS.model_file)
                    saver.restore(sess, FLAGS.model_file)

                    # 保证'generated'文件夹存在
                    generated_file = 'generated/' + name_list[i] + '_' + 'result.jpg'
                    if os.path.exists('generated') is False:
                        os.makedirs('generated')

                    # 产生和写入图片数据到文件
                    with open(generated_file, 'wb') as img:
                        start_time = time.time()
                        img.write(sess.run(tf.image.encode_jpeg(generated)))
                        end_time = time.time()
                        tf.logging.info('Elapsed time: %fs' % (end_time - start_time))

                        tf.logging.info('Done. Please check %s.' % generated_file)
        except Exception as err:
            print(err)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()

















