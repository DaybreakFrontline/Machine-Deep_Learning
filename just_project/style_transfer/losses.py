# import tensorflow as tf
import tensorflow._api.v2.compat.v1 as tf   # 将TensorFlow2.0转换为1.0
tf.disable_v2_behavior()

def gram(layer):
    shape = tf.shape(layer)
    num_images = shape[0]
    width = shape[1]
    height = shape[2]
    num_filters = shape[3]
    filters = tf.reshape(layer, tf.stack([num_images, -1, num_filters]))
    grams = tf.matmul(filters, filters, transpose_a=True) / tf.to_float(width * height * num_filters)

    return grams


# 定义内容损失
# endpoints_dict是损失网络各层的计算结果
# content_layers是定义使用哪些层的差距来计算损失，默认配置是conv3_3
def content_loss(endpoints_dict, content_layers):
    content_losses = 0
    for layer in content_layers:
        # 之前把生成图像、原始图像同时传入损失网络中计算
        # 这里需要先把它们区分开
        # tf.concat 和 tf.split
        generated_images, content_images = tf.split(endpoints_dict[layer], num_or_size_splits=2, axis=0)
        size = tf.size(generated_images)
        # 所谓的内容损失，是生成图片的激活generated_images与原始图片的激活content_images的L2距离
        content_losses += tf.nn.l2_loss(generated_images - content_images) * 2 / tf.to_float(size)

    return content_losses


# 定义风格损失
# endpoints_dict是损失网络各层的计算结果
# style_layers是定义使用哪些层计算风格损失。默认是conv1_2, conv2_2, conv3_3, conv4_3
# style_features_t是利用原始的风格图片计算的层的激活
# 例如在wave模型中是img/wave.jpg计算的激活
def style_loss(endpoints_dict, style_feature_t, style_layers):
    style_losses = 0
    # summary是为TensorBoard服务的
    style_loss_summary = {}
    for style_gram, layer in zip(style_feature_t, style_layers):
        # 计算风格损失，只需要计算生成图片generated_images与目标风格style_feature_t的差距。因此不需要取出content_images
        generated_images, _ = tf.split(endpoints_dict[layer], 2, 0)
        size = tf.size(generated_images)
        # 调用gram函数计算Gram矩阵。风格损失定义为生成图片与目标风格Gram矩阵的L2 loss
        layer_style_loss = tf.nn.l2_loss(gram(generated_images) - style_gram) * 2 / tf.to_float(size)
        style_loss_summary[layer] = layer_style_loss
        style_losses += layer_style_loss

    return style_losses, style_loss_summary



