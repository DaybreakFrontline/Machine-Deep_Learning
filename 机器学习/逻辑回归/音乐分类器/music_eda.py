from scipy.fft import fft
from scipy.io import wavfile
from matplotlib.pyplot import specgram
import matplotlib.pyplot as plt
import numpy as np

# 准备音乐数据，把音乐文件一个个的去使用傅里叶变换，并且把傅里叶变换之后的结果落地保存
# 提取特征  将每一首歌曲的傅里叶变换都存入本地文件


def create_fft(g, n):
    rad = "E:/pyCharmProject/Model_Data_Music/数据/genres/" + g + "/converted/" + g + '.'+str(n).zfill(5) + '.au.wav'
    sample_rate, X = wavfile.read(rad)
    # 事实证明，这个采样率设置高了只会分类失败
    fft_features = abs(fft(X)[:1200])
    sad = "E:/pyCharmProject/Model_Data_Music/trainset/" + g + '.' + str(n).zfill(5) + ".fft"
    np.save(sad, fft_features)


genre_list = ['classical', 'jazz', 'country', 'pop', 'rock', 'metal', 'blues', 'disco', 'hiphop', 'reggae']
for g in genre_list:
    for n in range(100):
        create_fft(g, n)


'''
(sample_rate, X) = wavfile.read("D:/genres/blues/converted/blues.00000.au.wav")
print(sample_rate, X.shape)

plt.figure(figsize=(10, 4), dpi=80)
plt.xlabel("time")
plt.ylabel("frequency")
plt.grid(True, linestyle='-', color='0.75')
specgram(X, Fs=sample_rate, xextent=(0, 30))
plt.show()


def plotSpec(g, n):
    sample_rate, X = wavfile.read("D:/genres/"+g+"/converted/"+g+'.'+n+'.au.wav')
    specgram(X, Fs=sample_rate, xextent=(0, 30))
    plt.title(g+"-"+n[-1])


plt.figure(num=None, figsize=(18, 9), dpi=80, facecolor='w', edgecolor='k')
plt.subplot(6, 3, 1); plotSpec("classical", '00001')
plt.subplot(6, 3, 2); plotSpec("classical", '00002')
plt.subplot(6, 3, 3); plotSpec("classical", '00003')
plt.subplot(6, 3, 4); plotSpec("jazz", '00001')
plt.subplot(6, 3, 5); plotSpec("jazz", '00002')
plt.subplot(6, 3, 6); plotSpec("jazz", '00003')
plt.subplot(6, 3, 7); plotSpec("country", '00001')
plt.subplot(6, 3, 8); plotSpec("country", '00002')
plt.subplot(6, 3, 9); plotSpec("country", '00003')
plt.subplot(6, 3, 10); plotSpec("pop", '00001')
plt.subplot(6, 3, 11); plotSpec("pop", '00002')
plt.subplot(6, 3, 12); plotSpec("pop", '00003')
plt.subplot(6, 3, 13); plotSpec("rock", '00001')
plt.subplot(6, 3, 14); plotSpec("rock", '00002')
plt.subplot(6, 3, 15); plotSpec("rock", '00003')
plt.subplot(6, 3, 16); plotSpec("metal", '00001')
plt.subplot(6, 3, 17); plotSpec("metal", '00002')
plt.subplot(6, 3, 18); plotSpec("metal", '00003')

plt.tight_layout(pad=0.4, w_pad=0, h_pad=1.0)
plt.show()


plt.figure(num=None, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')

plt.subplot(3, 2, 1)
sample_rate, a = wavfile.read("D:/StudyMaterials/python/python-sklearn/sine_a.wav")
plt.xlabel("time")
plt.ylabel("frequency")
plt.title('400 HZ sine wave')
specgram(a, Fs=sample_rate, xextent=(0, 30))
plt.subplot(3, 2, 2)
plt.xlabel("frequency")
plt.xlim((0, 1000))
plt.ylabel("amplitude")
plt.title('FFT of 400 HZ sine wave')
plt.plot(abs(fft(a, sample_rate)))

plt.subplot(3, 2, 3)
sample_rate, b = wavfile.read("D:/StudyMaterials/python/python-sklearn/sine_b.wav")
plt.xlabel("time")
plt.ylabel("frequency")
plt.title('3000 HZ sine wave')
specgram(b, Fs=sample_rate, xextent=(0, 30))
plt.subplot(3, 2, 4)
plt.xlabel("frequency")
plt.xlim((0, 4000))
plt.ylabel("amplitude")
plt.title('FFT of 3000 HZ sine wave')
plt.plot(abs(fft(b, sample_rate)))

plt.subplot(3, 2, 5)
sample_rate, mix = wavfile.read("D:/StudyMaterials/python/python-sklearn/sine_mix.wav")
plt.xlabel("time")
plt.ylabel("frequency")
plt.title('400 and 3000 HZ sine wave')
specgram(mix, Fs=sample_rate, xextent=(0, 30))
plt.subplot(3, 2, 6)
plt.xlabel("frequency")
plt.xlim((0, 4000))
plt.ylabel("amplitude")
plt.title('FFT of 400 and 3000 HZ sine wave')
plt.plot(abs(fft(mix, sample_rate)))

plt.show()


# sample_rate, X = wavfile.read("D:/genres/metal/converted/metal.00000.au.wav")
sample_rate, X = wavfile.read("D:/genres/pop/converted/pop.00000.au.wav")
plt.figure(num=None, figsize=(9, 6), dpi=80, facecolor='w', edgecolor='k')
plt.subplot(2, 1, 1)
plt.xlabel("time")
plt.ylabel("frequency")
specgram(X, Fs=sample_rate, xextent=(0, 30))
plt.subplot(2, 1, 2)
plt.xlabel("frequency")
plt.xlim((0, 3000))
plt.ylabel("amplitude")
plt.plot(fft(X, sample_rate))
plt.show()
'''
