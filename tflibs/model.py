"""
    Model
    ~~~~~
"""


class Model:
    @staticmethod
    def model_fn(features, labels, mode, params):
        raise NotImplementedError

    @classmethod
    def add_model_args(cls, argparser, parse_args):
        pass

    @classmethod
    def add_train_args(cls, argparser, parse_args):
        pass

    @classmethod
    def add_eval_args(cls, argparser, parse_args):
        pass


class Network:
    def __init__(self, is_chief, features, labels=None):
        self._is_chief = is_chief
        self._features = features
        self._labels = labels

    @property
    def is_chief(self):
        return self._is_chief

    @property
    def features(self):
        return self._features

    @property
    def labels(self):
        return self._labels
