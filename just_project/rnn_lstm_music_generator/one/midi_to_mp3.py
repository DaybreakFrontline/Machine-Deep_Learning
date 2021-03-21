import os
import subprocess


# midi转MP3的方法
def convert_midi_to_mp3(input_name):
    # 被转换的midi文件名字，默认output
    input_file = input_name + '.mid'
    # 转后的mp3文件名
    output_file = input_name + '_output.mp3'
    # 判断时候存在该文件
    if not os.path.exists(input_file):
        raise Exception("MIDI 文件 {} 不在此目录下，请确保此文件被正确生成".format(input_file))
    print('将 {} 转换为 MP3'.format(input_file))
    command = 'timidity {} -Ow -o - | ffmpeg -i - -acodec libmp3lame -ab 64k {}'.format(input_file, output_file)
    return_code = subprocess.call(command, shell=True)
    # 输出转换信息
    if return_code != 0:
        print('转换时出错，请手动转换或下载midi播放器')
    else:
        print('转换完毕. 生成的文件是 {}'.format(output_file))
