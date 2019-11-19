import math

import tensorflow as tf


class ReduceFunc(object):

    def apply(self, tensor):
        pass


class Mean(ReduceFunc):

    def apply(self, tensor):
        return tf.reduce_mean(input_tensor=tensor, axis=-2)  # reduce along sequence dimension


class ConvNet(ReduceFunc):
    def __init__(self, input_size, kernel_size, num_filters, max_length):
        """
        Initialize 1D CNN with max-over-time pooling reduction op.
        :param input_size: input/channel dimensionality
        :param kernel_size: size of 1D window, length of first dimension of each filter
        :param num_filters: number of [kernel_size, num_channels] filters, the dimensionality of output
        :param max_length: maximum length of 3rd dimension of input tensors
        """
        super(ConvNet, self).__init__()
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.filters = num_filters
        self.sequence_length = max_length

    def apply(self, tensor):
        return self.max_over_time_pooling_cnn(tensor=tensor, input_size=self.input_size,
                                              sequence_length=self.sequence_length,
                                              num_filters=self.filters, kernel_size=self.kernel_size)

    @staticmethod
    def max_over_time_pooling_cnn(tensor, input_size, num_filters, kernel_size, sequence_length=None):
        """
        Return a 1D CNN with max-over-time pooling.
        :param input_size: channel dimensionality
        :param tensor: 4D input tensor: [batch_size, time_steps, sequence_length, num_channels]
        :param num_filters: number of [kernel_size, num_channels] filters, the dimensionality of output
        :param kernel_size: size of 1D window, length of first dimension of each filter
        :param sequence_length: number of time steps (3rd dimension of input tensor)
        :return: 3D tensor [batch_size, time_steps, filters]
        """
        shape = tf.shape(input=tensor)
        if tensor.shape.ndims == 4:
            flatten = True
        elif tensor.shape.ndims == 3:
            flatten = False
        else:
            raise ValueError('Expecting 3 or 4-dimensional Tensor as input, got %s dims' % tensor.shape.ndims)
        # flatten sequences for input
        if flatten:
            tensor = tf.reshape(tensor, shape=[-1, sequence_length, input_size])

        limit = math.sqrt(3.0 / num_filters)
        initializer = tf.compat.v1.random_uniform_initializer(-limit, limit)
        tensor = tf.compat.v1.layers.conv1d(tensor, filters=num_filters, kernel_size=kernel_size, activation=tf.nn.relu,
                                  kernel_initializer=initializer)
        if flatten:
            tensor = tf.compat.v1.layers.max_pooling1d(tensor, pool_size=sequence_length - kernel_size + 1, strides=1)
            tensor = tf.reshape(tensor, shape=[-1, shape[1], num_filters])
        else:
            tensor = tf.reduce_max(input_tensor=tensor, axis=1)
        return tensor
