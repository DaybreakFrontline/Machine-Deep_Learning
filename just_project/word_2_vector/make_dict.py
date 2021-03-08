# 制作词表
import collections

from download_data import maybe_download
from load_data import read_data
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"     # 使用第一, 二块GPU

# 下载并取出语料库后，来制作一个单词表，它可以将单词映射为一个数字，这个数字是该单词的ID，即建立索引

# 一般来说，因为在语料库中有些词只出现有限的几次，如果单词表中包含了语料库中的所有词，会过于庞大。所以，
# 单词表一般只包含最常用的那些词。对于剩下的不常用的词，会将它替换为一个罕见词标记'UNK'，所有罕见的词都会
# 被映射为同一个单词ID

# 制作一个词表，将单词映射为一个的ID
# 词表的大小为5万，即只考虑最常出现的5万个词
# 将不常见的词变成一个UNK标识符，映射到统一的ID


def build_dataset(words, n_words):
    """
    将原始的单词表示变成index索引表示
    """
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words-1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # 如果没有的词就和UNK一样是索引0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


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


# 在这里的程序中，单词表中只包含了最常用的50000个单词。请注意，在这个实现中，名词的单复数形式，如boy和boys，
# 动词的不同时态，如make和made都被算作是不同的单词。原来的训练数据vocabulary是一个单词的列表，在经过转换后，
# 它变成了一个单词ID的列表，即程序中的变量data，它的形式是[5234, 3081, 12, 6, 195, 2, 3134, 46, ...]




