"""
    Dispatches operations over multiple GPU devices and calculates tower loss
"""

import tensorflow as tf

from tflibs.utils import map_dict, device_setter


class Dispatcher:
    """
    A class for dispatching operations over multiple GPU devices and caculating tower loss

    :param class model_cls: Model class.
    :param dict model_param: A dict of model parameters.
    :param list gpus: A list of gpu ids.
    :param dict features: A dict of `tf.Tensor` containing feature values.
    :param tf.Tensor labels: A `tf.Tensor` of labels.
    """

    def __init__(self, model_cls, model_param, gpus, features, labels=None):
        num_gpus = len(gpus)

        # Split inputs
        split_feature_dict = map_dict(lambda k, v: (k, tf.split(v, num_gpus)), features)
        split_features = map(lambda gpu: map_dict(lambda k, v: (k, v[gpu]), split_feature_dict), gpus)

        args = [gpus, split_features]
        if labels is not None:
            split_labels = tf.split(labels, num_gpus)
            args.append(split_labels)

        self._models = map(lambda args: model_cls(args[0] == 0,
                                                  *(args[1:]),
                                                  **model_param),
                           zip(*args))

    def minimize(self, optimizer, loss_fn, depends=None, global_step=None):
        """
        Gets `loss_fn` which maps model object to loss tensor, caculates tower loss and minimize it.

        :param tflibs.training.Optimizer optimizer: An optimizer.
        :param function loss_fn: A function that maps model object to loss tensor.
        :param tf.Operation depends: An operation that should be run before running an optimization.
        :return: Train op.
        :rtype: tf.Operation
        """
        if not isinstance(depends, (tuple, list)):
            depends = [depends] if depends is not None else None
        with tf.control_dependencies(depends):
            def compute_grad((device_id, model)):
                with tf.device(device_setter('/gpu:{}'.format(device_id))):
                    loss = loss_fn(model)
                    return optimizer.compute_grad(loss)

            tower_grads = map(compute_grad, enumerate(self._models))
            apply_grad_op = optimizer.apply_tower_gradients(tower_grads, global_step=global_step)

            return apply_grad_op

    @property
    def models(self):
        return self._models

    @property
    def chief(self):
        return filter(lambda model: model.is_chief, self.models)[0]
