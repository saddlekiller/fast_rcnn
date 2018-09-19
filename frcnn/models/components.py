import tensorflow as tf
import numpy as np
from frcnn.models.modules import *
from frcnn.utils.utils import *
from PIL import Image

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
                            self.layers['stack_conv_{}'.format(i+1)]['relu_{}'.format(j + 1)] = ReluActivation()
                    self.layers['stack_conv_{}'.format(i+1)]['maxpooling'] = MaxPooling2D(pool_size=(2, 2),
                                                                                          strides=(2, 2),
                                                                                          padding='VALID')

            self.layers['stack_dense'] = {}
            with tf.variable_scope('fc6'):
                self.layers['stack_dense']['dense_6'] = Conv2D7x7(in_channels=512,
                                             out_channels=4096, padding='VALID')
            with tf.variable_scope('fc7'):
                self.layers['stack_dense']['dense_7'] = Conv2D1x1(in_channels=4096,
                                             out_channels=4096)
            with tf.variable_scope('fc8'):
                self.layers['stack_dense']['dense_8'] = Conv2D1x1(in_channels=4096,
                                             out_channels=1000)

    def load(self, ckpt_path, sess):
        saver = tf.train.Saver()
        saver.restore(sess, ckpt_path)

    def __call__(self, inputs):
        x = inputs - self.mean_rgb / 255
        for stack_name, stack_layers in self.layers.items():
            for layer_name, layer in stack_layers.items():
                x = layer(x)
        return tf.nn.top_k(tf.squeeze(x, [2, 2]), k = 3)[1]


class ResNet():

    def __init__(self):
        pass

    def __call__(self, inputs):
        pass


if __name__ == '__main__':
    # print_model('../pretrain/vgg_16.ckpt', all_tensors=True)
    img = Image.open('../tests/images/n02423022_0.JPEG')
    img = img.resize((224, 224))
    img = np.array(img) / 255
    img = np.expand_dims(img, 0)
    print(np.array(img).shape)

    with tf.Session() as sess:
        vgg = VGG16()
        vgg.load('../pretrain/vgg_16.ckpt', sess)
        results = sess.run(vgg(tf.constant(img, dtype=tf.float32)))[0]
        print(results[0])
        # print(sess.run(vgg.mean_rgb))
        # print(sess.run(vgg.global_step))
