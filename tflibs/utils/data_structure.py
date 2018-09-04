"""
    Utils
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def _prepend(key, dic):
    items = list(map(lambda item: ('{}/{}'.format(key, item[0]), item[1]), dic.items()))
    return dict(items)


def flatten_nested_dict(nested_dict):
    flatten_dict = dict(_nested_dict_item_gen(nested_dict))
    return flatten_dict


def _nested_dict_item_gen(nested_dict):
    for (k, v) in nested_dict.items():
        if not isinstance(v, dict):
            yield (k, v)
        else:
            for (inner_k, inner_v) in _nested_dict_item_gen(v):
                yield ('{}/{}'.format(k, inner_k), inner_v)


def map_dict(map_fn, original_dict):
    return dict(map(lambda item: map_fn(item[0], item[1]), original_dict.items()))
