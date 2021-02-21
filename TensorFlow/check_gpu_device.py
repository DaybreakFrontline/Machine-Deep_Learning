import tensorflow as tf
from tensorflow.python.client import device_lib

print(tf.__version__)
print(tf.__path__)

print('=============================')

print(tf.test.gpu_device_name())
print(tf.test.is_gpu_available())
print('=============================')

# 列出所有的本地机器设备
local_device_protos = device_lib.list_local_devices()
# 打印
print(local_device_protos)
print('=============================')

# 只打印GPU设备
[print(x) for x in local_device_protos if x.device_type == 'GPU']

print(tf.config.list_physical_devices('GPU'))