from just_project.rnn_lstm_music_generator.one import midi_to_mp3
from music21 import converter, instrument, note, chord


def create_music(prediction):
    """
    用神经网络'预测'的音乐数据来生成 MIDI 文件，再转成 MP3 文件
    """
    offset = 0  # 偏移
    output_notes = []

    # 生成 Note（音符）或 Chord（和弦）对象
    for data in prediction:
        # 是 Chord。格式例如： 4.15.7
        if ('.' in data) or data.isdigit():
            notes_in_chord = data.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()  # 乐器用钢琴 (piano)
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # 是 Note
        else:
            new_note = note.Note(data)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        # 每次迭代都将偏移增加，这样才不会交叠覆盖
        offset += 0.5

    # 创建音乐流（Stream）
    midi_stream = stream.Stream(output_notes)

    # 写入 MIDI 文件
    midi_stream.write('midi', fp='output.mid')

    # 将生成的 MIDI 文件转换成 MP3
    midi_to_mp3.convert_midi_to_mp3('testMusic')
