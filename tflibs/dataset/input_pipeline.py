"""
    Input pipeline
"""
import itertools

import tensorflow as tf

from tflibs.utils import compose_funcs
from tflibs.ops import randomly_pick_op


def augment_data_op(augment_fns: list):
    with tf.name_scope('augment_data'):
        def binary_counter(length):
            return itertools.product(range(2), repeat=length)

        def binary_selection(col):
            return [itertools.compress(col, list(bin)) for bin in binary_counter(len(col))]

        augs = [compose_funcs(*selected) for selected in binary_selection(augment_fns)]

        aug = randomly_pick_op(augs)

        return aug
