import tensorflow as tf

class Conv2D():

    def __init__(self, ksize, in_channels, out_channels, padding='SAME', use_bias=True):
        if type(ksize) == int:
            weight_shape = (ksize, ksize, in_channels, out_channels)
        else:
            weight_shape = (ksize[0], ksize[1], in_channels, out_channels)

        self.use_bias = use_bias
        self.weights = tf.get_variable(name='weights',
                                       initializer=tf.truncated_normal_initializer(),
                                       shape=weight_shape)
        self.padding = padding
        if self.use_bias:
            self.biases = tf.get_variable(name='biases',
                                          initializer=tf.zeros_initializer(),
                                          shape=(out_channels))

    def __call__(self, inputs, strides=(1, 1, 1, 1)):
        outputs = tf.nn.conv2d(input=inputs,
                               filter=self.weights,
                               strides=strides,
                               padding=self.padding)
        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.biases)
        return outputs


class Conv2D3x3(Conv2D):

    def __init__(self, in_channels, out_channels, padding='SAME', use_bias=True):
        super(Conv2D3x3, self).__init__(3, in_channels, out_channels, padding, use_bias)

class Conv2D7x7(Conv2D):

    def __init__(self, in_channels, out_channels, padding='SAME', use_bias=True):
        super(Conv2D7x7, self).__init__(7, in_channels, out_channels, padding, use_bias)

class Conv2D1x1(Conv2D):

    def __init__(self, in_channels, out_channels, padding='SAME', use_bias=True):
        super(Conv2D1x1, self).__init__(1, in_channels, out_channels, padding, use_bias)

class Dense():

    def __init__(self, in_dim, out_dim, use_bias=True):
        self.use_bias = use_bias
        self.weights = tf.get_variable(name='weights',
                                      initializer=tf.truncated_normal_initializer(),
                                      shape=(in_dim, out_dim))
        if self.use_bias:
            self.biases = tf.get_variable(name='biases',
                                          initializer=tf.zeros_initializer(),
                                          shape=(out_dim))

    def __call__(self, inputs):
        outputs = tf.matmul(inputs, self.weights)
        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.biases)
        return outputs

class MaxPooling2D():

    def __init__(self, pool_size, strides, padding):
        self.pooling = tf.layers.MaxPooling2D(pool_size=pool_size, strides=strides, padding=padding)

    def __call__(self, inputs):
        return self.pooling(inputs)

class ReluActivation():

    def __init__(self):
        pass

    def __call__(self, inputs):
        return tf.nn.relu(inputs)