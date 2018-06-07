""" Model """


class Model:
    @staticmethod
    def model_fn(features, labels, mode, params):
        raise NotImplementedError

    @classmethod
    def add_model_args(cls, argparser, parse_args):
        raise NotImplementedError

    @classmethod
    def add_train_args(cls, argparser, parse_args):
        raise NotImplementedError

    @classmethod
    def add_eval_args(cls, argparser, parse_args):
        raise NotImplementedError
