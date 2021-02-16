import numpy as np
import pickle
from pprint import pprint
from scipy.io import wavfile
from scipy.fft import fft

genre_list = ['classical', 'jazz', 'country', 'pop', 'rock', 'metal', 'blues', 'disco', 'hiphop', 'reggae']


pkl_file = open('model.pkl', 'rb')
model_loaded = pickle.load(pkl_file)
pprint(model_loaded)
pkl_file.close()

print('Starting read wavfile...')
music_name = '下川直哉 - キミガタメ_劇伴.wav'
sample_rate, X = wavfile.read("E:/pyCharmProject/Model_Data_Music/sample/" + music_name)

print(X.shape)
X = np.reshape(X, (1, -1))[0]

print(X.shape)
print(sample_rate, X)

# 事实证明，这个采样率设置高了只会分类失败
test_fft_features = abs(fft(X)[:1200])
print(sample_rate, test_fft_features, len(test_fft_features))

result_index = model_loaded.predict([test_fft_features])[0]
print(music_name + " 的风格是: " + genre_list[result_index])
