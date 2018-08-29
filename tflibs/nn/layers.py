import tensorflow as tf


# TODO: Enum constants
class Padding:
    Constant = 'CONSTANT'
    Reflect = 'REFLECT'
    Symmetric = 'SYMMETRIC'
    NONE = 'NONE'


class Norm:
    Batch = tf.layers.batch_normalization
    Instance = tf.contrib.layers.instance_norm
    Layer = tf.contrib.layers.layer_norm
    NONE = None


class Nonlinear:
    ReLU = tf.nn.relu
    LeakyReLU = tf.nn.leaky_relu
    Sigmoid = tf.nn.sigmoid
    Tanh = tf.nn.tanh
    NONE = None


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
            padding = [((kernel_size - 1) * dilation_rate + 1 - strides) / 2] * 2
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
        if norm_fn:
            inputs = norm_fn(inputs)

        # Non-linearity
        if non_linear_fn:
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
                inputs = tf.image.resize_nearest_neighbor(inputs, map(lambda e: e * strides, shape))

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
            # Padding
            if padding_mode.upper() != Padding.NONE:
                padding = [((kernel_size - 1) + 1 - strides) / 2] * 2
                inputs = tf.pad(inputs, [[0] * 2, padding, padding, [0] * 2], mode=padding_mode)

            inputs = tf.layers.conv2d_transpose(inputs,
                                                num_filters,
                                                kernel_size,
                                                strides=(strides, strides),
                                                padding='VALID',
                                                use_bias=use_bias)

            # Normalization
            if norm_fn:
                inputs = norm_fn(inputs)

            # Non-linearity
            if non_linear_fn:
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
                   scope=None,
                   reuse=None):
    with tf.variable_scope(scope, 'Residual_Block', values=[inputs], reuse=reuse):
        shortcut = inputs
        inputs = conv2d(inputs, num_filters, 3, padding_mode=padding_mode, norm_fn=norm_fn, non_linear_fn=non_linear_fn)
        inputs = conv2d(inputs, num_filters, 3, padding_mode=padding_mode, norm_fn=norm_fn, non_linear_fn=None)

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
        if norm_fn:
            inputs = norm_fn(inputs)

        # Non-linearity
        if non_linear_fn:
            inputs = non_linear_fn(inputs)

        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, inputs)

        return inputs


def adaptive_instance_norm(inputs, gamma, beta, scope=None):
    with tf.variable_scope(scope, 'Adaptive_Instance_Norm', values=[inputs, gamma, beta]):
        inputs = tf.contrib.layers.instance_norm(inputs, center=False, scale=False)
        inputs = gamma * inputs + beta

        return inputs