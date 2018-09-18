import tensorflow as tf
from frcnn.models.modules import *
import numpy as np

def test_Conv2D3x3():
    shape = (2, 20, 30, 10)
    in_channels = shape[-1]
    out_channels = 5
    inputs = tf.constant(np.random.random(shape), dtype=tf.float32)
    conv2d3x3 = Conv2D3x3(in_channels, out_channels)
    outputs = conv2d3x3(inputs)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run([outputs])[0].shape)

if __name__ == '__main__':
    test_Conv2D3x3()
