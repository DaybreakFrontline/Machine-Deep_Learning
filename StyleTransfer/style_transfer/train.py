import tensorflow._api.v2.compat.v1 as tf   # 将TensorFlow2.0转换为1.0
tf.disable_v2_behavior()
from StyleTransfer.style_transfer.vgg import vgg_16
from StyleTransfer.style_transfer.vgg import vgg_arg_scope
from StyleTransfer.style_transfer.vgg_preprocessing import preprocess_image
from StyleTransfer.style_transfer.vgg_preprocessing import unprocess_image
from StyleTransfer.style_transfer import reader
from StyleTransfer.style_transfer import model
from StyleTransfer.style_transfer import utils
from StyleTransfer.style_transfer.losses import gram
from StyleTransfer.style_transfer import losses
import os
import time
import tf_slim as slim
# slim = tf.contrib.slim


# 构建损失网络
# tensorboard --logdir='style_transfer\models\wave'

# 对图片做预处理
def get_preprocessing(name, is_training=False):
    # 返回一个函数，处理每个批次里面的每张图像
    # image = preprocessing_fn(image, output_height, output_width, ...).
    def preprocessing_fn(image, output_height, output_width, **kwargs):
        return preprocess_image(image, output_height, output_width, is_training=is_training, **kwargs)

    def unprocessing_fn(image):
        return unprocess_image(image)

    return preprocessing_fn, unprocessing_fn


# 得到经典网络VGG16
def get_network_fn(name, num_classes, weight_decay=0.0, is_training=False):
    arg_scope = vgg_arg_scope(weight_decay=weight_decay)

    def network_fn(images, **kwargs):
        with slim.arg_scope(arg_scope):
            return vgg_16(images, num_classes, is_training=is_training, **kwargs)

    if hasattr(vgg_16, 'default_image_size'):
        network_fn.defualt_image_size = vgg_16.default_image_size

    return network_fn


def build_loss_network(FLAGS):

    # 得到损失网络
    network_fn = get_network_fn('vgg16', num_classes=1, is_training=False)

    # 获取数据预处理函数
    image_preprocessing_fn, image_unprocess_fn = get_preprocessing('vgg16')
    # 读入图片顺便做数据预处理
    processed_images = reader.image(FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 'train2017/'
                                    , image_preprocessing_fn, epochs=FLAGS.epoch)

    # 引用图像生成网络model.net，generated是生成的图像
    # 设置training=True因为要训练该网络
    generated = model.net(processed_images, training=True)

    # 将生成的图像同样使用image_preprocessing_fn进行处理
    # 因为generated同样需要送到损失网络中计算loss
    processed_generated = [
        image_preprocessing_fn(image, FLAGS.image_size, FLAGS.image_size)
        for image in tf.unstack(generated, axis=0, num=FLAGS.batch_size)
    ]
    processed_generated = tf.stack(processed_generated)

    # 这里需要原始的经过预处理的图像，和原始的经过生成网络再经过预处理的图像
    # 参数标志是否对输出进行squeeze操作（去除维度数为1的维度，比如5*3*1转为5*3）
    _, endpoints = network_fn(tf.concat([processed_generated, processed_images], 0), spatial_squeeze=False)

    return endpoints


# 计算出来风格图片的特征
def get_style_features(FLAGS):
    with tf.Graph().as_default():
        network_fn = get_network_fn('vgg16', num_classes=1, is_training=False)
        # 获取数据预处理函数
        image_preprocessing_fn, image_unprocessing_fn = get_preprocessing('vgg16')

        # 获取风格图片
        size = FLAGS.image_size
        img_bytes = tf.read_file(FLAGS.style_image)
        if FLAGS.style_image.lower().endswith('png'):
            image = tf.image.decode_png(img_bytes)
        else:
            image = tf.image.decode_jpeg(img_bytes)

        # 增加批次维度
        images = tf.expand_dims(image_preprocessing_fn(image, size, size), 0)
        # 让这张风格图片经过网络正向传播
        _, endpoints_dict = network_fn(images, spatial_squeeze=False)

        features = []
        for layer in FLAGS.style_layers:
            feature = endpoints_dict[layer]
            # 去掉批次的维度
            feature = tf.squeeze(gram(feature), [0])
            features.append(feature)

        with tf.Session() as sess:
            # 恢复复原变量为了损失网络
            init_func = utils._get_init_fn(FLAGS)
            init_func(sess)

            # 保证'generated'字典存在
            if os.path.exists('generated') is False:
                os.makedirs('generated')
            # 指出剪裁的风格图片路径
            save_file = 'generated/target_style_' + FLAGS.naming + '.jpg'
            # 写入预处理的风格图片到指定的路径
            with open(save_file, 'wb') as f:
                target_image = image_unprocessing_fn(images[0, :])
                value = tf.image.encode_jpeg(tf.cast(target_image, tf.uint8))
                f.write(sess.run(value))
                tf.logging.info('Target style pattern is saved to: %s.' % save_file)

            # 返回那些用来计算风格损失的网络层的特征
            return sess.run(features)


def main(FLAGS):
    # 返回那些用来计算风格损失的网络层的特征，先计算出来备用
    style_features_t = get_style_features(FLAGS)

    # 保证训练路径存在
    training_path = os.path.join(FLAGS.model_path, FLAGS.naming)
    if not(os.path.exists(training_path)):
        os.makedirs(training_path)

    with tf.Graph().as_default():
        with tf.Session() as sess:
            # 获取数据预处理函数
            image_preprocessing_fn, image_unprocessing_fn = get_preprocessing('vgg16')
            # 获取训练所需图片
            processed_images = reader.image(FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size,
                                            'train2017/', image_preprocessing_fn, epochs=FLAGS.epoch)
            # 经过卷积反卷积的生成网络的图片
            generated = model.net(processed_images, training=True)
            processed_generated = [image_preprocessing_fn(image, FLAGS.image_size, FLAGS.image_size)
                                   for image in tf.unstack(generated, axis=0, num=FLAGS.batch_size)
                                   ]
            processed_generated = tf.stack(processed_generated)
            # 构建网络
            network_fn = get_network_fn('vgg16', num_classes=1, is_training=False)
            _, endpoints_dict = network_fn(tf.concat([processed_generated, processed_images], 0), spatial_squeeze=False)

            # 记录损失网络的结构
            tf.logging.info('Loss network layers(You can define them in "content_layers" and "style_layers"):')
            for key in endpoints_dict:
                tf.logging.info(key)

            # 构建损失
            content_loss = losses.content_loss(endpoints_dict, FLAGS.content_layers)
            style_loss, style_loss_summary = losses.style_loss(endpoints_dict, style_features_t, FLAGS.style_layers)
            loss = FLAGS.style_weight * style_loss + FLAGS.content_weight * content_loss

            # 添加一些总结为了tensorboard可视化
            tf.summary.scalar('losses/content_loss', content_loss)
            tf.summary.scalar('losses/style_loss', style_loss)

            tf.summary.scalar('weighted_losses/weighted_content_loss', content_loss * FLAGS.content_weight)
            tf.summary.scalar('weighted_losses/weighted_style_loss', style_loss * FLAGS.style_weight)
            tf.summary.scalar('total_loss', loss)

            for layer in FLAGS.style_layers:
                tf.summary.scalar('style_losses/' + layer, style_loss_summary[layer])
            tf.summary.image('generated', generated)
            tf.summary.image('origin', tf.stack([
                image_unprocessing_fn(image) for image in tf.unstack(processed_images, axis=0, num=FLAGS.batch_size)
            ]))
            summary = tf.summary.merge_all()
            writer = tf.summary.FileWriter(training_path)

            # 准备去训练
            global_step = tf.Variable(0, name='global_step', trainable=False)

            variable_to_train = []
            for variable in tf.trainable_variables():
                if not(variable.name.startswith(FLAGS.loss_model)):
                    variable_to_train.append(variable)

            train_op = tf.train.AdamOptimizer(1e-3).minimize(loss, global_step=global_step, var_list=variable_to_train)

            variable_to_restore = []
            for v in tf.global_variables():
                if not(v.name.startswith(FLAGS.loss_model)):
                    variable_to_restore.append(v)

            saver = tf.train.Saver(variable_to_restore, write_version=tf.train.SaverDef.V1)

            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

            # 恢复损失网络的变量
            init_func = utils._get_init_fn(FLAGS)
            init_func(sess)

            # 恢复对于模型训练的变量如果checkpoint文件已经存在
            last_file = tf.train.latest_checkpoint(training_path)
            if last_file:
                tf.logging.info('Restoring model from {}'.format(last_file))
                saver.restore(sess, last_file)

            # 开始训练
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            start_time = time.time()
            try:
                while not coord.should_stop():
                    _, loss_t, step = sess.run([train_op, loss, global_step])
                    elapsed_time = time.time() - start_time
                    start_time = time.time()
                    # 记录日志
                    if step % 5 == 0:
                        tf.logging.info('step: %d, total Loss %f, secs/step: %f' % (step, loss_t, elapsed_time))
                    # 总结
                    if step % 25 == 0:
                        tf.logging.info('adding summary...')
                        summary_str = sess.run(summary)
                        writer.add_summary(summary_str, step)
                        writer.flush()
                    if step % 1000 == 0:
                        saver.save(sess, os.path.join(training_path, 'fast-style-model.ckpt'), global_step=step)
                    if step % 10000 == 0:
                        sed_time = time.time() - start_time
                        start_time = time.time()
                        tf.logging.info('step: %d, total Loss %f, 一万次耗时: %f' % (step, loss_t, sed_time))
            except tf.errors.OutOfRangeError:
                saver.save(sess, os.path.join(training_path, 'fast-style-model.ckpt-done'))
                tf.logging.info('Done training -- epoch limit reached')
            finally:
                coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    FLAGS = utils.read_conf_file('conf/wave.yml')
    main(FLAGS)



















