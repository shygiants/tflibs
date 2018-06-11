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
