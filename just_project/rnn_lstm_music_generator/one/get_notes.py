import os
import glob
import pickle
from music21 import converter, instrument, note, chord


def get_notes():
    # 读取所有midi文件
    if not os.path.exists("K:\music_midi"):
        raise Exception("包含所有 MIDI 文件的 music_midi 文件夹不在此目录下，请添加")

    notes = []
    count = 0
    # glob : 匹配所有符合条件的文件，并以 List 的形式返回
    # glob模块是最简单的模块之一，内容非常少。用它可以查找符合特定规则的文件路径名。
    for midi_file in glob.glob("K:\music_midi\*.mid"):
        print(midi_file)
        # music21.converter.parse方法
        stream = converter.parse(midi_file)

        # nstrument.partitionByInstrument(stream) 获取所有乐器部分。
        parts = instrument.partitionByInstrument(stream)

        if parts:  # 如果有乐器部分，取第一个乐器部分，否则难度太大了
            notes_to_parse = parts.parts[0].recurse()
        else:
            notes_to_parse = stream.flat.notes

        # 取音调，若是和弦，则转成音符
        for element in notes_to_parse:
            # 如果是 Note 类型，那么取它的音调
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            # 如果是 Chord 类型，那么取它各个音调的序号
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
        count += 1
        print('正在处理第:', count, '个midi文件')
    # 保存所取的音调
    # 如果 data 目录不存在，创建此目录
    if not os.path.exists("data"):
        os.mkdir("data")
    # 将数据写入 data 目录下的 notes 文件
    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)
    return notes


if __name__ == '__main__':
    get_notes()
