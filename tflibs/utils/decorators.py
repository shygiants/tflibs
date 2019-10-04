""" Decorators """
from typing import Callable, TypeVar
from functools import wraps

import tensorflow as tf

T = TypeVar('T')


class CachedProperty:
    def __init__(self, fget: Callable):
        self.fget = fget

    @property
    def fget(self) -> Callable:
        return self._fget

    @fget.setter
    def fget(self, fn):
        self._fget = fn
        self._name = fn.__name__

    @property
    def name(self) -> str:
        return self._name

    def __get__(self, instance, owner) -> T:
        """
        Called once when the corresponding property is first accessed, and sets the value as attribute.
        :param instance:
        :param owner:
        :return: Value of the property
        """
        val = self.fget(instance)  # type: T

        setattr(instance, self.name, val)

        return val


def coroutine(f: Callable):
    @wraps(f)
    async def wrapper(*args, **kwargs):
        return f(*args, **kwargs)

    return wrapper


def distributed_run(strategy: tf.distribute.MirroredStrategy, run_eager=False):
    def decorator(f: Callable):
        @wraps(f)
        def wrapper(*args, **kwargs):
            return strategy.experimental_run_v2(f, args=args, kwargs=kwargs)

        return tf.function(wrapper) if not run_eager else wrapper

    return decorator


def unpack_tuple(original_fn: Callable):
    @wraps(original_fn)
    def wrapper(tup):
        return original_fn(*tup)

    return wrapper


def unpack_dict(original_fn: Callable):
    @wraps(original_fn)
    def wrapper(dic):
        return original_fn(**dic)

    return wrapper
