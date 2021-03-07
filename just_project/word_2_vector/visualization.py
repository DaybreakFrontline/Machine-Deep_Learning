import matplotlib.pyplot as plt


# 由于之前设定的embedding_size=128，即每个词都被表示为一个128维的向量，虽然没有方法把128维的空间
# 直接画出来，但是下面的程序使用了t-SNE方法把128维空间映射到了2维，并画出最常使用的500个词的位置。
# 画出的图片保存为tsne.png文件

def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(20, 20))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.savefig(filename)
    # 因为循环太多，每个循环里面画了一张图，所以可以在每个循环内把plt关闭
    plt.close()

