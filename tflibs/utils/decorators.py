""" Decorators """

import tensorflow as tf


def name_scope(original_fn=None, name=None):
    def decorator(original_fn):
        def wrapper(*args, **kwargs):
            with tf.name_scope(name or original_fn.__name__):
                return original_fn(*args, **kwargs)

        return wrapper

    if original_fn is not None:
        if name is not None:
            raise ValueError('Bad usage')

        return decorator(original_fn)
    else:
        return decorator


def strip_dict_arg(original_fn):
    def wrapper(arg):
        return original_fn(**arg)

    return wrapper