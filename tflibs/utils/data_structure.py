"""
    Utils
"""
import functools


class Attributes:
    def __init__(self, **attrs):
        self.update(**attrs)
        self._attrs = attrs

    def update(self, **attrs):
        for name, value in attrs.items():
            setattr(self, name, value)

    def __getattr__(self, item):
        raise AttributeError

    @property
    def __dict__(self):
        return self._attrs


class LambdaAttributes:
    def __init__(self, **attrs):
        self.update(**attrs)

    def update(self, **attrs):
        for name, value in attrs.items():
            setattr(self, '_' + name, value)

    def __getattr__(self, item: str):
        if item.startswith('_'):
            raise AttributeError

        func = getattr(self, '_' + item)
        val = func()
        setattr(self, item, val)

        return val


def _prepend(key, dic):
    items = list(map(lambda item: ('{}/{}'.format(key, item[0]), item[1]), dic.items()))
    return dict(items)


def flatten_nested_dict(nested_dict: dict) -> dict:
    flatten_dict = dict(_nested_dict_item_gen(nested_dict))
    return flatten_dict


def _nested_dict_item_gen(nested_dict: dict):
    for (k, v) in nested_dict.items():
        if not isinstance(v, dict):
            yield (k, v)
        else:
            for (inner_k, inner_v) in _nested_dict_item_gen(v):
                yield ('{}/{}'.format(k, inner_k), inner_v)


def param_consumer(arg_names, params, unpack=False):
    values = map(lambda n: params.pop(n), arg_names)
    if unpack:
        return list(values)
    else:
        return dict(zip(arg_names, values))


def compose_funcs(*funcs):
    def compose(f, g):
        return lambda x: f(g(x))

    return functools.reduce(compose, funcs, lambda x: x)


def list_enum(enum):
    return list(map(lambda e: e.value, list(enum)))
