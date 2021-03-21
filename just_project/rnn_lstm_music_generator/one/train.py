import tensorflow as tf
from just_project.rnn_lstm_music_generator.one import get_notes
from just_project.rnn_lstm_music_generator.one import network_model_rnn as network_model
from just_project.rnn_lstm_music_generator.one import prepare_sequences


# 训练神经网络
def train():
    notes = get_notes.get_notes()

    # 得到所有不重复（因为用了 set）的音调数目
    num_pitch = len(set(notes))

    network_input, network_output = prepare_sequences.prepare_sequences(notes, num_pitch)

    model = network_model.network_model(network_input, num_pitch)
    # 官方定义的格式
    filepath = "model\weights.{epoch:02d}-{loss:.4f}.hdf5"

    # 用 Checkpoint（检查点）文件在每一个 Epoch 结束时保存模型的参数（Weights）
    # 不怕训练过程中丢失模型参数。可以在我们对 Loss（损失）满意了的时候随时停止训练
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath,  # 保存的文件路径
        monitor='loss',  # 监控的对象是 损失（loss）
        verbose=0,
        save_best_only=True,  # 不替换最近的数值最佳的监控对象的文件
        mode='min'  # 取损失最小的
    )
    callbacks_list = [checkpoint]

    # 用 fit 方法来训练模型
    model.fit(network_input, network_output, epochs=100, batch_size=64, callbacks=callbacks_list)
    score = model.evaluate(network_input)
    print(score)


if __name__ == '__main__':
    train()