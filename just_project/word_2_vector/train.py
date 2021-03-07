import tensorflow as tf
import numpy as np
from sklearn.manifold import TSNE
from make_trainset import generate_batch
from define_model import generate_graph
from download_data import maybe_download
from load_data import read_data
from make_dict import build_dataset
from visualization import plot_with_labels


def train(graph, init, train_inputs, train_labels, loss, optimizer, normalized_embeddings, similarity
          , reverse_dictionary, data):

    num_steps = 100001

    with tf.Session(graph=graph) as session:
        # 初始化变量
        init.run()
        print('Initialized')

        average_loss = 0
        for step in range(num_steps):
            batch_inputs, batch_labels = generate_batch(
                batch_size=batch_size, num_skips=num_skips, skip_window=skip_window, data=data
            )
            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

            # 优化一下
            _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
            average_loss += loss_val

            if step % 2000 == 0:
                if step > 0:
                    average_loss /= 2000
                # 2000个batch的平均损失
                print('Average loss at step ', step, ': ', average_loss)
                average_loss = 0

            # 每1万步，进行一次验证
            if step % 10000 == 0:
                # sim是验证词与所有词之间的相似度
                sim = similarity.eval()
                # 一共有valid_size个验证词
                for i in range(valid_size):
                    valid_word = reverse_dictionary[valid_examples[i]]
                    top_k = 8  # 输出最相邻的8个词语
                    nearest = (-sim[i, :]).argsort()[1: top_k+1]
                    log_str = 'Nearest to %s:' % valid_word
                    for k in range(top_k):
                        close_word = reverse_dictionary[nearest[k]]
                        log_str = '%s %s,' % (log_str, close_word)
                    print(log_str)
                # final_embeddings是最后得到的embedding向量
                # 它的形状是[vocabulary_size, embedding_size]
                # 每一行代表着对应单词id的词嵌入表示
                final_embeddings = normalized_embeddings.eval()
                # 最终，得到的词嵌入向量为final_embeddings，它是归一化后的词嵌入向量，
                # 形状为(vocabulary_size, embedding_size)，final_embeddings[0, :]
                # 是id为0的单词对应的词嵌入表示，final_embeddings[1, :]是id为1的单词
                # 对应的词嵌入表示，以此类推

                # 因为embedding大小为128维，没有办法直接可视化
                # 所以用t-SNE方法进行降维
                tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
                # 只画出500个词的位置
                plot_only = 500
                low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
                labels = [reverse_dictionary[i] for i in range(plot_only)]
                plot_with_labels(low_dim_embs, labels)


if __name__ == '__main__':
    url = 'http://mattmahoney.net/dc/'
    filename = maybe_download(url, 'text8.zip', 31344016)
    vocabulary = read_data(filename=filename)
    vocabulary_size = 50000
    data, count, dictionary, reverse_dictionary = build_dataset(vocabulary, vocabulary_size)
    del vocabulary  # 删除以节省内存
    # 输出最常见的5个单词
    print('Most common words (+UNK)', count[:5])
    # 输出转换后的数据库data，和原来的单词，前10个
    print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

    # 在训练过程中，会对模型进行验证
    # 验证的方法是找出和某个词最近的词
    # 只对前valid_window的词进行验证，因为这些词最常出现
    valid_size = 16  # 每次验证16个词
    valid_window = 100  # 这16个词是从前100个最常见的词中选出来的
    valid_examples = np.random.choice(valid_window, valid_size, replace=False)

    batch_size = 128
    skip_window = 1  # skip_window参数和之前保持一致
    num_skips = 2  # num_skips参数和之前保持一致

    graph, init, train_inputs, train_labels, loss, optimizer, normalized_embeddings, similarity = \
        generate_graph(vocabulary_size, valid_examples)

    train(graph, init, train_inputs, train_labels, loss, optimizer, normalized_embeddings
          , similarity, reverse_dictionary, data)

