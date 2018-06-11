""" Logging """

import tensorflow as tf


def log_parse_args(parse_args, argument_class='Arguments'):
    arg_dict = parse_args.__dict__
    arg_str = reduce(lambda a, b: a + b, map(lambda (k, v): '{}: {}\n'.format(k, v), arg_dict.iteritems()), '')
    tf.logging.info('{}: %s'.format(argument_class), arg_str)
