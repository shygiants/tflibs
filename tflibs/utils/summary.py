""" Summary """

import tensorflow as tf


def strip_illegal_summary_name(name):
    """
    Strips illegal summary name
    :param str name:
    :return: Stripped name
    """
    return name.rstrip(':0')


def histogram_trainable_vars():
    trainable_vars = tf.trainable_variables()
    for var in trainable_vars:
        tf.summary.histogram(strip_illegal_summary_name(var.name), var, family='Trainable_Variables')


def image_summary(name, image):
    batch_size = image.shape.as_list()[0]
    tf.summary.image(name, image, max_outputs=batch_size, family='Images')
