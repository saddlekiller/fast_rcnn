import tensorflow as tf
import numpy as np
from frcnn.models.modules import *
from frcnn.utils.utils import *
from PIL import Image
from frcnn.dataset.feeder import TinyImageNetFeeder, VOCFeeder

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
        self.layers = {}
        self.global_step = tf.get_variable(name='global_step', shape=(), dtype=tf.int64)
        self.layers_names = []

        with tf.variable_scope('vgg_16'):
            self.mean_rgb = tf.get_variable(name='mean_rgb', shape=(3), dtype=tf.float32)
            for i in range(len(self.stack_filters) - 1):
                self.layers['stack_conv_{}'.format(i+1)] = {}
                with tf.variable_scope('conv{}'.format(i+1)):
                    if i == 0 or i == 1:
                        n_layers = 2
                    else:
                        n_layers = 3
                    for j in range(n_layers):
                        with tf.variable_scope('conv{}_{}'.format(i+1, j+1)):
                            if j == 0:
                                self.layers['stack_conv_{}'.format(i+1)]['conv_{}'.format(j+1)] = Conv2D3x3(in_channels=self.stack_filters[i],
                                                            out_channels=self.stack_filters[i+1])
                            else:
                                self.layers['stack_conv_{}'.format(i + 1)]['conv_{}'.format(j + 1)] = Conv2D3x3(in_channels=self.stack_filters[i+1],
                                                            out_channels=self.stack_filters[i+1])
                            self.layers_names.append('stack_conv_{}'.format(i + 1) + '/conv_{}'.format(j + 1))
                            self.layers['stack_conv_{}'.format(i+1)]['relu_{}'.format(j + 1)] = ReluActivation()
                    self.layers['stack_conv_{}'.format(i+1)]['maxpooling'] = MaxPooling2D(pool_size=(2, 2),
                                                                                          strides=(2, 2),
                                                                                          padding='VALID')
                    self.layers_names.append('stack_conv_{}'.format(i+1) + '/maxpooling')


            self.layers['stack_dense'] = {}
            with tf.variable_scope('fc6'):
                self.layers['stack_dense']['dense_6'] = Conv2D7x7(in_channels=512,
                                             out_channels=4096, padding='VALID')
                self.layers_names.append('stack_dense/dense_6')
            with tf.variable_scope('fc7'):
                self.layers['stack_dense']['dense_7'] = Conv2D1x1(in_channels=4096,
                                             out_channels=4096)
                self.layers_names.append('stack_dense/dense_7')
            with tf.variable_scope('fc8'):
                self.layers['stack_dense']['dense_8'] = Conv2D1x1(in_channels=4096,
                                             out_channels=1000)
                self.layers_names.append('stack_dense/dense_8')
        logger.hline()
        logger.info(' __     __  _____    _____  ')
        logger.info('| |    / / / ___ \  / ___ \ ')
        logger.info('| |   / / / /  /_/ / /  /_/ ')
        logger.info('| |  / / / / ____ / / ____  ')
        logger.info('| | / / / / /  _// / /  _/  ')
        logger.info('| |/ / / /__/ / / /__/ /    ')
        logger.info('|___/  \_____/  \_____/     ')
        logger.hline()
        for l in self.layers_names:
            scope, name = l.split('/')
            logger.info('{}: {}'.format(l, self.layers[scope][name]))
        logger.hline()

    def load(self, ckpt_path, sess):
        saver = tf.train.Saver()
        saver.restore(sess, ckpt_path)

    # def __call__(self, inputs):
    #     x = inputs - self.mean_rgb
    #     for stack_name, stack_layers in self.layers.items():
    #         for layer_name, layer in stack_layers.items():
    #             x = layer(x)
    #     return tf.nn.top_k(tf.squeeze(x, [2, 2]), k = 3)[1]
        # return tf.argmax(tf.squeeze(x, [2, 2]), axis=-1)

    def __call__(self, inputs, output_layer_name):
        if output_layer_name not in self.layers_names:
            logger.error('{} not found in VGG-16')
            raise ValueError
        x = inputs - self.mean_rgb
        for l in self.layers_names:
            scope, name = l.split('/')
            x = self.layers[scope][name](x)
            if l == output_layer_name:
                break
        return x

class ResNet():

    def __init__(self):
        pass

    def __call__(self, inputs):
        pass



if __name__ == '__main__':
    from frcnn.configs.hparams import hparams
    placeholder = tf.placeholder(tf.float32, (None, None, None, 3))
    vgg = VGG16()
    output = vgg(placeholder, 'stack_conv_5/conv_3')
    print(output)
    feeder = VOCFeeder(hparams.train_preproc)
    # print(data[0].shape)
    import matplotlib.pyplot as plt

    for i in range(10):
        data = feeder._prepare_batch(1)
        img = data[0][0]
        box = data[2]
        filename = data[-1][0]
        print(img.shape, box)
        print(filename)
        annotate_image(img, box, True, True)
        plt.show()