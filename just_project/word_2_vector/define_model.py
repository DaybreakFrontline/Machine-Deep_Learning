import tensorflow._api.v2.compat.v1 as tf   # 将TensorFlow2.0转换为1.0
tf.disable_v2_behavior()
import math

# 此处的模型实际可以抽象为：用一个单词预测另一个单词，在输出时，不使用softmax损失，而使用NCE损失，
# 即再选取一些"噪声词"，作为负样本进行两类分类
# 建立模型

def generate_graph(vocabulary_size, valid_examples):
    batch_size = 128
    embedding_size = 128  # 词嵌入空间是128维的。即word2vec中的vec是一个128维的向量

    # 构造损失时选取的噪声词的数量
    num_sampled = 64

    graph = tf.Graph()
    with graph.as_default():
        # 输入的batch
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

        # 用于验证的词
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

        # 下面采用的某些函数还没有GPU实现，所以只在CPU上定义模型
        with tf.device('/cpu:0'):
            # 定义一个embeddings变量，这个变量的形状是（vocabulary_size,embedding_size）
            # 相当于每一行存储了一个单词的嵌入向量embedding，例如，单词id为0的嵌入是
            # embeddings[0,:]，单词id为1的嵌入是embeddings[1:]，依次类推
            embeddings = tf.Variable(
                tf.random_uniform([vocabulary_size, embedding_size], -1.0, -1.0)
            )
            # 利用embedding_lookup可以轻松得到一个batch内的所有的词嵌入
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)

            # 创建两个变量用于NCE Loss（即选取噪声词的二分类损失）
            nce_weights = tf.Variable(
                tf.truncated_normal([vocabulary_size, embedding_size],
                                    stddev=1.0 / math.sqrt(embedding_size))
            )
            nce_bias = tf.Variable(tf.zeros([vocabulary_size]))

        # tf.nn.nce_loss会自动选取噪声词，并且形成损失
        # 随机选取num_sampled个噪声词
        loss = tf.reduce_mean(
            tf.nn.nce_loss(weights=nce_weights, biases=nce_bias,
                           labels=train_labels, inputs=embed,
                           num_sampled=num_sampled, num_classes=vocabulary_size)
        )
        # 得到loss后，可以构造优化器了
        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

        # 对embedding层做一次归一化
        # 由于直接得到的embeddings矩阵可能在各个维度上有不同的大小，为了使计算的相似度更合理，
        # 先对其做一次归一化，用归一化后的normalized_embeddings计算验证词和其他单词的相似度。
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        # 找出和验证词的embedding并计算它们和所有单词的相似度（用于验证）
        # 在训练模型时，还希望对模型进行验证。此处采取的方法是选出一些"验证单词"，
        # 计算在嵌入空间中与其最相近的词。
        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
        similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

        # 变量初始化步骤
        init = tf.global_variables_initializer()

    return graph, init, train_inputs, train_labels, loss, optimizer, normalized_embeddings, similarity
