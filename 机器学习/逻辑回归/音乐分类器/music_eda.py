# 对数据的预处理 Explore Data Analysis

from scipy import fft  # 快速傅里叶变换
from scipy.io import wavfile  # 读取wav格式的文件
from matplotlib.pyplot import specgram  # 绘制频谱图
import matplotlib.pyplot as plt

# sample_rate 采样率 采样个数  赫兹为单位   计算机每秒采集多少信号样本
(sample_rate, X) = wavfile.read("Q:/尚学堂人工智能/百战人工智能33期1组/08_逻辑回归2：推广到多分类，实战音乐分类器"
                                "/数据/genres/blues/converted/blues.00000.au.wav")

# X.shape 如果是一元组，那么他就是个单通道的， 目前是一元组
print(sample_rate, X.shape)  # 音乐文件大概的长度就是 X.shape / sample_rate 秒

# 创建画布
plt.figure(figsize=(10, 4), dpi=120)  # 画布尺寸
plt.xlabel("time")  # x轴
plt.ylabel("frequency")  # 频率
plt.grid(True, linestyle='-', color='0.75')  # 绘制网格
specgram(X, Fs=sample_rate, xextent=(0, 120))  # Fs采样率     xextent时间长度
# plt.show()


def plotSpec(g, n):
    sample_rate, X = wavfile.read("Q:/尚学堂人工智能/百战人工智能33期1组/08_逻辑回归2：推广到多分类，实战音乐分类器"
                                  "/数据/genres/" + g + "/converted/" + g + '.' + n + '.au.wav')
    specgram(X, Fs=sample_rate, xextent=(0, 30))
    plt.title(g + "_" + n[-1])


plt.figure(num=None, figsize=(18, 9), dpi=100, facecolor='w', edgecolor='k')  # 设置画布
plt.subplot(6, 3, 1);   plotSpec("classical", '00001');
plt.subplot(6, 3, 2);   plotSpec("classical", '00002');
plt.subplot(6, 3, 3);   plotSpec("classical", '00003');
plt.subplot(6, 3, 4);   plotSpec("jazz", '00001');
plt.subplot(6, 3, 5);   plotSpec("jazz", '00002');
plt.subplot(6, 3, 6);   plotSpec("jazz", '00003');
plt.subplot(6, 3, 7);   plotSpec("country", '00001');
plt.subplot(6, 3, 8);   plotSpec("country", '00002');
plt.subplot(6, 3, 9);   plotSpec("country", '00003');

plt.subplot(6, 3, 10);   plotSpec("pop", '00001');
plt.subplot(6, 3, 11);   plotSpec("pop", '00002');
plt.subplot(6, 3, 12);   plotSpec("pop", '00003');
plt.subplot(6, 3, 13);   plotSpec("rock", '00001');
plt.subplot(6, 3, 14);   plotSpec("rock", '00002');
plt.subplot(6, 3, 15);   plotSpec("rock", '00003');
plt.subplot(6, 3, 16);   plotSpec("metal", '00001');
plt.subplot(6, 3, 17);   plotSpec("metal", '00002');
plt.subplot(6, 3, 18);   plotSpec("metal", '00003');

plt.tight_layout(pad=0.4, w_pad=0, h_pad=1.0)
plt.show()
