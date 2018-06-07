""" Logging """

import tensorflow as tf


def log_parse_args(parse_args):
    arg_dict = parse_args.__dict__
    arg_str = reduce(lambda a, b: a + b, map(lambda (k, v): '{}: {}\n'.format(k, v), arg_dict.iteritems()))
    tf.logging.info('Arguments: %s', arg_str)
