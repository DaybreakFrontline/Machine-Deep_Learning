# 下载、验证完成后，使用下面的程序将语料库中的数据读出来
import zipfile
import tensorflow._api.v2.compat.v1 as tf   # 将TensorFlow2.0转换为1.0
tf.disable_v2_behavior()

from download_data import maybe_download


# 将语料库解压，并转换成一个word的list
def read_data(filename):
    """
    将下载好的zip文件解压并读取为word的list
    """
    with zipfile.ZipFile(filename) as f:
        print(f.namelist()[0])
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()

    return data


if __name__ == '__main__':
    url = 'http://mattmahoney.net/dc/'
    filename = maybe_download(url, 'text8.zip', 31344016)
    vocabulary = read_data(filename=filename)
    print('Data size', len(vocabulary))
    # 输出前100个词
    # 词语本来是在连续的句子中的，现在已经被去掉了标点
    print(vocabulary[0:100])








