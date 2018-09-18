import tensorflow as tf
import numpy as np
from frcnn.models.modules import *
from frcnn.utils.utils import *

class SPP():

    def __init__(self):
        pass

    def __call__(self, inputs):
        pass

class RPN():

    def __init__(self):
        pass

    def __call__(self, inputs):
        pass

class VGG19():

    def __init__(self):
        pass

    def __call__(self, inputs):
        pass

class VGG16():

    def __init__(self):
        self.stack_filters = [3, 64, 128, 256, 512, 512]
        self.stack_denses = [512, 4096, 4096, 1000]
        self.layers = []
        with tf.variable_scope('vgg_16'):
            for i in range(len(self.stack_filters) - 1):
                with tf.variable_scope('conv{}'.format(i+1)):
                    if i == 0 or i == 1:
                        n_layers = 2
                    else:
                        n_layers = 3
                    for j in range(n_layers):
                        with tf.variable_scope('conv{}_{}'.format(i+1, j+1)):
                            if j == 0:
                                self.layers.append(Conv2D3x3(in_channels=self.stack_filters[i],
                                                            out_channels=self.stack_filters[i+1]))
                            else:
                                self.layers.append(Conv2D3x3(in_channels=self.stack_filters[i+1],
                                                            out_channels=self.stack_filters[i+1]))
            with tf.variable_scope('fc6'):
                self.layers.append(Conv2D7x7(in_channels=512,
                                             out_channels=4096))
            with tf.variable_scope('fc7'):
                self.layers.append(Conv2D1x1(in_channels=4096,
                                             out_channels=4096))
            with tf.variable_scope('fc8'):
                self.layers.append(Conv2D1x1(in_channels=4096,
                                             out_channels=1000))

    def load(self, ckpt_path, sess):
        saver = tf.train.Saver()
        saver.restore(sess, ckpt_path)

    def __call__(self, inputs):
        x = inputs

class ResNet():

    def __init__(self):
        pass

    def __call__(self, inputs):
        pass


if __name__ == '__main__':

    with tf.Session() as sess:
        vgg = VGG16()
        vgg.load('../pretrain/vgg_16.ckpt', sess)


