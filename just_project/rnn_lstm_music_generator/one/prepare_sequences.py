import numpy as np
import tensorflow as tf


# 为神经网络准备好供训练的序列
def prepare_sequences(notes, num_pitch):
    sequence_length = 100  # 序列长度
    # 得到所有音调的名字
    pitch_names = sorted(set(item for item in notes))

    # 创建一个字典，用于映射 音调 和 整数
    pitch_to_int = dict((pitch, num) for num, pitch in enumerate(pitch_names))

    # 创建神经网络的输入序列和输出序列
    network_input = []
    network_output = []

    for i in range(0, len(notes) - sequence_length, 1):
        # 每次取sequence_length个音符
        sequence_in = notes[i: i + sequence_length]
        # sequence_length个音符推出的一个音符
        sequence_out = notes[i + sequence_length]
        # 更新序列
        network_input.append([pitch_to_int[char] for char in sequence_in])
        network_output.append(pitch_to_int[sequence_out])
    n_patterns = len(network_input)
    # 将输入的形状转换成神经网络模型可以接受的
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    # 将 输入 标准化 / 归一化
    # 归一话可以让之后的优化器（optimizer）更快更好地找到误差最小值
    network_input = network_input / float(num_pitch)

    # 将期望输出转换成 {0, 1} 组成的布尔矩阵，为了配合 categorical_crossentropy 误差算法使用
    network_output = tf.keras.utils.to_categorical(network_output)

    return network_input, network_output
