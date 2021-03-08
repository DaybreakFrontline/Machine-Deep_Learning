# 导入一些需要的库

# 由于Python是由社区推动的开源并且免费的开发语言，不受商业公司控制，因此，Python的改进往往比较激进，
# 不兼容的情况时有发生。Python为了确保你能顺利过渡到新版本，特别提供了__future__模块，
# 让你在旧的版本中试验新版本的一些特性。

# 如果你在main.py中写import string,那么在Python 2.4或之前, Python会先查找当前目录下
# 有没有string.py, 若找到了，则引入该模块，然后你在main.py中可以直接用string了。
# 如果你是真的想用同目录下的string.py那就好，但是如果你是想用系统自带的标准string.py呢？
# 那其实没有什么好的简洁的方式可以忽略掉同目录的string.py而引入系统自带的标准string.py。
# 这时候你就需要from __future__ import absolute_import了。这样，你就可以用
# import string来引入系统的标准string.py, 而用from pkg import string来引入
# 当前目录下的string.py了
from __future__ import absolute_import

# 如果你想在Python 2.7的代码中直接使用Python 3.x的精准除法，可以通过__future__模块的division实现
from __future__ import division
from __future__ import print_function

import math
import os
from six.moves import urllib
from six.moves import xrange

# 为了使用Skip-Gram方法训练语言模型，需要下载对应语言的语料库。在网站http://mattmahoney.net/dc/
# 上提供了大量英语语料库下载，为了方便学习，使用一个比较小的语料库http://mattmahoney.net/dc/text8.zip
# 作为示例训练模型，程序会自动下载这个文件


def maybe_download(url, filename, expected_bytes):
    """
    :param filename: 如果filename不存在，在上面的地址下载它
    :param expected_bytes: 如果filename存在，跳过下载
    最终会检查文字的字节数是否和expected_bytes相同
    """
    if not os.path.exists(filename):
        filename, _ = urllib.request.urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception(
            'Failed to verify ' + filename + '. Can you get to it with a browser?'
        )
    return filename


if __name__ == '__main__':
    url = 'http://mattmahoney.net/dc/'
    filename = maybe_download(url, 'text8.zip', 31344016)

# 如果读者运行这段程序后，发现没有办法正常下载文件，可以尝试使用URL手动下载，并将下载好的文件放在当前目录下







