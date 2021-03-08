import random
import numpy as np
import collections
from download_data import maybe_download
from load_data import read_data
from make_dict import build_dataset
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"     # 使用第一, 二块GPU

# 得到的变量data包含了训练集中所有的数据，现在把它转换成训练时使用的batch数据
# 一个batch可以看作是一些"单词对"的集合，如woman->man，woman->fell，箭头左边表示"出现的单词"，
# 右边表示该单词所在的"上下文"中的单词，这是所说的Skip-Gram方法
data_index = 0


def generate_batch(batch_size, num_skips, skip_window, data):
    """
    每运行一次这个函数，会产生一个batch的数据以及对应的标签labels
    :param batch_size: 一个批次中单词对的个数
    :param num_skips: 在生成单词对时，会在语料库中先取一个长度为skip_window * 2 + 1连续单词列表
                    这个单词列表放在上面程序中的变量buffer。buffer中最中间的那个单词是skip-gram
                    方法中"出现的单词"，其余的skip_window * 2个单词是它的"上下文"。
                    会在skip_window*2个单词中随机选取num_skips个单词，放入标签labels
    :param skip_window:
    :param data:
    :return: 返回两个值batch和labels，前者表示skip-gram方法中"出现的单词"，后者表示"上下文"中的单词
            它们的形状分别为（batch_size,）和 （batch_size, 1）
    """
    # data_index相当于一个指针，初始为0
    # 每次生成一个batch，data_index会相应地往后推
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window

    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)

    # data_index是当前数据开始的位置
    # 产生batch后往后推1位（产生batch）
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        # 利用buffer生成batch
        # buffer是一个长度为2*skip_window + 1长度的word list
        # 一个buffer生成num_skips个数的样本
        target = skip_window  # target label at the center of the buffer
        # targets_to_avoid保证样本不重复
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        # 每利用buffer生成num_skips个样本，data_index向后推进一位
        data_index = (data_index + 1) % len(data)
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels


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

    # 默认情况下skip_window=1, num_skips=2
    # 此时是从连续的3 （3 = skip_window * 2 + 1）个词中生成2(num_skips)个样本
    # 如连续的三个词['used', 'against', 'early']
    # 生成两个样本：against -> used, against -> early
    batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1, data=data)
    for i in range(8):
        print(batch[i], reverse_dictionary[batch[i]], '->', labels[i, 0], reverse_dictionary[labels[i, 0]])



























