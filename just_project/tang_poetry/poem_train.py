import collections
import numpy as np
import tensorflow as tf

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



