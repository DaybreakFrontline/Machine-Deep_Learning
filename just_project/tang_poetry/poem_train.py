import collections
import numpy as np
import tensorflow._api.v2.compat.v1 as tf

# 数据预处理 ================================================

poetry_file = './data/poetry.txt'

# 诗集
poetrys = []
with open(poetry_file, "r", encoding='utf-8', ) as f:
    print(f)
    for line in f:
        try:
            title, content = line.strip().split(':')
            content = content.replace(' ', '')
            if '_' in content or '(' in content or '（' in content or ' 《' in content or '[' in content:
                continue
            if len(content) < 5 or len(content) > 79:
                continue
            content = '[' + content + ']'
            poetrys.append(content)
        except Exception as e:
            pass

# 按诗的字数排序   reverse=False 正序  reverse .v 颠倒；彻底转变；使完全相反
poetrys = sorted(poetrys, key=lambda line : len(line), reverse=False)
print('唐诗总数:', len(poetrys))

# 统计每个字出现的次数
all_words = []  # 这里边的字是有重复的
for poetry in poetrys:
    temp = [word for word in poetry]
    all_words += temp

# 统计all_words数组里边元素的个数
counter = collections.Counter(all_words)    # 里边是个键值对 key：元素， value是出现次数
print(counter.items())
# 排序，按照键值对的【1】进行排序，就是按照元素的出现次数排序， reverse=True 进行倒排
count_pairs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
print(count_pairs)
# * 把列表里的元素取出来
print(*count_pairs)
# words 得到一个词表， _是所有的出现次数，我们不需要，所以用_代替
words, _ = zip(*count_pairs)
# 倒序排序后取出了所有字
print(words)
print(len(words))

# 取出一些常用的字，取前3001个常用字，并且加上空格
print(len(words))
words = words[:3000] + (' ',)
print(words)
print(len(words))

# 每个字映射为一个数字ID  len(words):01 - 6019, 在跟现有的words进行zip,再做成字典
word_num_map = dict(zip(words, range(len(words))))
print(word_num_map)

# 把诗转换为向量形式
# 定义一个查索引的方式，如果是常用字就给index，如果不是就给默认值len(words)
to_num= lambda word: word_num_map.get(word, len(words))
poetrys_vector = [list(map(to_num, poetry)) for poetry in poetrys]

# 每次取256首诗进行训练
batch_size = 256
# 计算多少次可以把古诗学完  向下取整
n_chunk = len(poetrys_vector) // batch_size
# 准备数据
x_batches = []
y_batches = []
for i in range(n_chunk):
    start_index = i * batch_size    # 开始的时候 i = 0   start_index = 0
    end_index = start_index + batch_size    # 0 - batch_size 就是 0 - 256
    # 每次取256首诗
    batches = poetrys_vector[start_index:end_index]
    # 计算256首诗中最长的长度
    length = max(map(len, length))
    # 创建全部为空格的索引号的矩阵    batch_size行 length列，
    xdata = np.full((batch_size, length), word_num_map[' '], np.int32)
    # 把每首诗的向量填入
    for row in range(batch_size):
        xdata[row, :len(batches[row])] = batches[row]
    ydata = np.copy(xdata)
    ydata[:, :-1] = xdata[:, 1:]

    # xdata             ydata
    # [6,2,4,6,9]     [2,4,6,9,9]
    # [1,4,2,8,5]     [4,2,8,5,5]

    x_batches.append(xdata)
    y_batches.append(ydata)

# -----------------------------------  RNN  -----------------------------
input_data = tf.placeholder(tf.int32, [batch_size, None])
output_targets = tf.placeholder(tf.int32, [batch_size, None])


# 定义RNN 默认model='lstm'
def neural_network(model='lstm', rnn_size=128, num_layer=2):
    if model == 'rnn':
        cell_fun = tf.nn.rnn_cell.BasicRNNCell
    elif model == 'gru':
        cell_fun = tf.nn.rnn_cell.GRUCell       # 比LSTMCell结构再简单一些，效率更高一些，只有一点点
    elif model == 'lstm':
        cell_fun = tf.nn.rnn_cell.BasicLSTMCell
    # 这时候才是调用对应的类，创建实例，上边接的不是实例、 rnn_size:隐藏神经元个数、 state_is_tuple=Ture 隐藏层结果放在tuple返回
    cell = cell_fun(rnn_size, state_is_tuple=True)

    # 单个节点里面神经网络有两层，堆叠的，相当于网络层更深、 [cell] * num_layer 放入的元素X2、 【10】* 2 = 【10, 10】
    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layer, state_is_tuple=True)     # [cell, cell] 传给 MultiRNNCell

    initial_state = cell.zero_state(batch_size, tf.float32) # 全部初始化为0


