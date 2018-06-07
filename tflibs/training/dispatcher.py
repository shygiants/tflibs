""" Dispatcher """

import tensorflow as tf

from tflibs.utils import map_dict, device_setter


class Dispatcher:
    def __init__(self, model_cls, model_param, gpus, features, labels):
        num_gpus = len(gpus)

        # Split inputs
        split_feature_dict = map_dict(lambda k, v: (k, tf.split(v, num_gpus)), features)
        split_features = map(lambda gpu: map_dict(lambda k, v: (k, v[gpu]), split_feature_dict), gpus)
        split_labels = tf.split(labels, num_gpus)
        self._models = map(lambda (gpu, features, labels): model_cls(is_chief=gpu == 0,
                                                                     features=features,
                                                                     labels=labels,
                                                                     **model_param),
                           zip(gpus, split_features, split_labels))

    def minimize(self, optimizer, loss_fn, depends=None):
        if not isinstance(depends, (tuple, list)):
            depends = [depends] if depends is not None else None
        with tf.control_dependencies(depends):
            def compute_grad((device_id, model)):
                with tf.device(device_setter('/gpu:{}'.format(device_id))):
                    loss = loss_fn(model)
                    return optimizer.compute_grad(loss)

            tower_grads = map(compute_grad, enumerate(self._models))
            apply_grad_op = optimizer.apply_tower_gradients(tower_grads)

            return apply_grad_op

    @property
    def models(self):
        return self._models

    @property
    def chief(self):
        return filter(lambda model: model.is_chief, self.models)[0]
