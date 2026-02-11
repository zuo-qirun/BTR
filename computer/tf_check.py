import tensorflow as tf
import sys
print('Python:', sys.version.replace('\n',' '))
print('TensorFlow version:', tf.__version__)
print('Physical GPUs:', tf.config.list_physical_devices('GPU'))
print('All physical devices:', tf.config.list_physical_devices())
print('Logical devices:', tf.config.list_logical_devices())
print('TF built with CUDA:', tf.test.is_built_with_cuda())
try:
    from tensorflow.python.client import device_lib
    print('device_lib.list_local_devices():')
    print(device_lib.list_local_devices())
except Exception as e:
    print('device_lib error:', e)
