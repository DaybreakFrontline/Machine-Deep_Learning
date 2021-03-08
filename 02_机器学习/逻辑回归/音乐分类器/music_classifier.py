import numpy as np
import pickle
from pprint import pprint
from scipy.io import wavfile
from scipy.fft import fft
from sklearn.linear_model import LogisticRegression

genre_list = ['classical', 'jazz', 'country', 'pop', 'rock', 'metal', 'blues', 'disco', 'hiphop', 'reggae']

# 读取傅里叶变换之后的数据集，将其做成机器学习所需要的X和y
X = []
y = []
for g in genre_list:
    for n in range(100):
        rad = "E:/pyCharmProject/Model_Data_Music/trainset/" + g + "." + str(n).zfill(5)+".fft.npy"
        fft_features = np.load(rad)
        X.append(fft_features)
        y.append(genre_list.index(g))

X = np.array(X)
y = np.array(y)

# 训练模型并且保存模型 multi_class='multinomial', solver='sag',
# 这个地方的迭代次数啊，设置成自动感觉不太好，收敛不了，如果设置次数太大，那就过拟合了
model = LogisticRegression(max_iter=3500)
model.fit(X, y)
output = open('model.pkl', 'wb')
pickle.dump(model, output)
output.close()
