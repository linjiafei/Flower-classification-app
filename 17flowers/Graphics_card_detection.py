'''
import tensorflow as tf

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")
'''

''''''
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
print(tf.__version__)
a = tf.constant(1.)
b = tf.constant(2.)
print(a+b)
print('GPU:', tf.test.is_gpu_available())


