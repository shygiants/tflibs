from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


# TODO: Enum constants
class Padding:
    Constant = 'CONSTANT'
    Reflect = 'REFLECT'
    Symmetric = 'SYMMETRIC'
    NONE = 'NONE'


class Norm:
    Batch = 'Batch'
    Instance = 'Instance'
    Layer = 'Layer'
    NONE = None

    @classmethod
    def get(cls, name):
        if name == cls.Batch:
            return tf.layers.batch_normalization
        elif name == cls.Instance:
            return tf.contrib.layers.instance_norm
        elif name == cls.Layer:
            return tf.contrib.layers.layer_norm
        else:
            raise ValueError()


class Nonlinear:
    ReLU = 'ReLU'
    LeakyReLU = 'LeakyReLU'
    Sigmoid = 'Sigmoid'
    Tanh = 'Tanh'
    NONE = None

    @classmethod
    def get(cls, name):
        if name == cls.ReLU:
            return tf.nn.relu
        elif name == cls.LeakyReLU:
            return tf.nn.leaky_relu
        elif name == cls.Sigmoid:
            return tf.nn.sigmoid
        elif name == cls.Tanh:
            return tf.nn.tanh
        else:
            raise ValueError()


class DeconvMethod:
    NNConv = 'NNConv'
    ConvTranspose = 'ConvTranspose'


def conv2d(inputs,
           num_filters,
           kernel_size,
           strides=1,
           dilation_rate=1,
           padding_mode=Padding.Constant,
           norm_fn=Norm.Instance,
           non_linear_fn=Nonlinear.ReLU,
           use_bias=True,
           scope=None,
           reuse=None):
    with tf.variable_scope(scope, 'Conv2d_{kernel_size}x{kernel_size}_{num_filters}'.format(kernel_size=kernel_size,
                                                                                            num_filters=num_filters),
                           values=[inputs], reuse=reuse):
        # Padding
        if padding_mode.upper() != Padding.NONE:
            padding = kernel_size + (kernel_size - 1) * (dilation_rate - 1) - strides
            padding = [padding // 2, padding - padding // 2]
            inputs = tf.pad(inputs, [[0] * 2, padding, padding, [0] * 2], mode=padding_mode)

        # Conv
        inputs = tf.layers.conv2d(inputs,
                                  num_filters,
                                  kernel_size,
                                  strides=(strides, strides),
                                  dilation_rate=(dilation_rate, dilation_rate),
                                  padding='VALID',
                                  use_bias=use_bias)

        # Normalization
        if norm_fn is not None:
            if not callable(norm_fn):
                norm_fn = Norm.get(norm_fn)
            inputs = norm_fn(inputs)

        # Non-linearity
        if non_linear_fn is not None:
            if not callable(non_linear_fn):
                non_linear_fn = Nonlinear.get(non_linear_fn)
            inputs = non_linear_fn(inputs)

        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, inputs)

        return inputs


def deconv2d(inputs,
             num_filters,
             kernel_size,
             strides=1,
             padding_mode=Padding.Constant,
             norm_fn=Norm.Layer,
             non_linear_fn=Nonlinear.ReLU,
             use_bias=True,
             method=DeconvMethod.NNConv,
             scope=None,
             reuse=None):
    with tf.variable_scope(scope, 'Deconv2d_{kernel_size}x{kernel_size}_{num_filters}'.format(kernel_size=kernel_size,
                                                                                              num_filters=num_filters),
                           values=[inputs], reuse=reuse):
        if method == DeconvMethod.NNConv:
            if strides > 1:
                shape = inputs.shape.as_list()[1:3]
                inputs = tf.image.resize_nearest_neighbor(inputs, list(map(lambda e: e * strides, shape)))

            # Conv
            inputs = conv2d(inputs,
                            num_filters,
                            kernel_size,
                            strides=1,
                            padding_mode=padding_mode,
                            norm_fn=norm_fn,
                            non_linear_fn=non_linear_fn,
                            use_bias=use_bias)
        elif method == DeconvMethod.ConvTranspose:
            inputs = tf.layers.conv2d_transpose(inputs,
                                                num_filters,
                                                kernel_size,
                                                strides=(strides, strides),
                                                padding='SAME',
                                                use_bias=use_bias)

            # Normalization
            if norm_fn is not None:
                if not callable(norm_fn):
                    norm_fn = Norm.get(norm_fn)
                inputs = norm_fn(inputs)

            # Non-linearity
            if non_linear_fn is not None:
                if not callable(non_linear_fn):
                    non_linear_fn = Nonlinear.get(non_linear_fn)
                inputs = non_linear_fn(inputs)

        else:
            raise ValueError('`method` should be either {} or {}'.format(DeconvMethod.NNConv,
                                                                         DeconvMethod.ConvTranspose))

        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, inputs)

        return inputs


def residual_block(inputs,
                   num_filters,
                   padding_mode=Padding.Constant,
                   norm_fn=Norm.Instance,
                   non_linear_fn=Nonlinear.ReLU,
                   use_bias=True,
                   scope=None,
                   reuse=None):
    with tf.variable_scope(scope, 'Residual_Block', values=[inputs], reuse=reuse):
        shortcut = inputs
        inputs = conv2d(inputs, num_filters, 3,
                        padding_mode=padding_mode, norm_fn=norm_fn, non_linear_fn=non_linear_fn, use_bias=use_bias)
        inputs = conv2d(inputs, num_filters, 3,
                        padding_mode=padding_mode, norm_fn=norm_fn, non_linear_fn=Nonlinear.NONE, use_bias=use_bias)

        return inputs + shortcut


def linear(inputs,
           num_units,
           norm_fn=Norm.Instance,
           non_linear_fn=Nonlinear.ReLU,
           use_bias=True,
           scope=None,
           reuse=None):
    with tf.variable_scope(scope, 'Linear_{num_units}'.format(num_units=num_units),
                           values=[inputs], reuse=reuse):
        # FC
        inputs = tf.layers.dense(inputs,
                                 num_units,
                                 use_bias=use_bias)

        # Normalization
        if norm_fn is not None:
            if not callable(norm_fn):
                norm_fn = Norm.get(norm_fn)
            inputs = norm_fn(inputs)

        # Non-linearity
        if non_linear_fn is not None:
            if not callable(non_linear_fn):
                non_linear_fn = Nonlinear.get(non_linear_fn)
            inputs = non_linear_fn(inputs)

        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, inputs)

        return inputs


def adaptive_instance_norm(inputs, gamma, beta, scope=None):
    with tf.variable_scope(scope, 'Adaptive_Instance_Norm', values=[inputs, gamma, beta]):
        inputs = tf.contrib.layers.instance_norm(inputs, center=False, scale=False)
        inputs = gamma * inputs + beta

        return inputs
