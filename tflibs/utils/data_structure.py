"""
    Utils
"""


def _prepend(key, dic):
    items = map(lambda (k, v): ('{}/{}'.format(key, k), v), dic.iteritems())
    return dict(items)


def flatten_nested_dict(nested_dict):
    flatten_dict = dict(_nested_dict_item_gen(nested_dict))
    return flatten_dict


def _nested_dict_item_gen(nested_dict):
    for (k, v) in nested_dict.iteritems():
        if not isinstance(v, dict):
            yield (k, v)
        else:
            for (inner_k, inner_v) in _nested_dict_item_gen(v):
                yield ('{}/{}'.format(k, inner_k), inner_v)


def map_dict(map_fn, original_dict):
    return dict(map(lambda (k, v): map_fn(k, v), original_dict.iteritems()))
