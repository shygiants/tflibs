"""
    Optimizer
"""

import tensorflow as tf

from tflibs.utils import strip_illegal_summary_name


# TODO: Universal optimizer
class Optimizer:
    def __init__(self, learning_rate, var_scope, beta1, beta2, decay_policy='none', decay_params=None):
        if decay_policy == 'none':
            self._learning_rate = learning_rate
        elif decay_policy == 'dying':
            self._learning_rate = self.dying_decay(learning_rate, **decay_params)
        elif decay_policy == 'step':
            self._learning_rate = self.step_decay(learning_rate, **decay_params)
        else:
            raise ValueError('`decay_policy` should be `none`, `dying` or `step`.')

        tf.summary.scalar(var_scope, self._learning_rate, family='Learning_Rates')

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate, beta1=beta1, beta2=beta2)
        self.var_scope = var_scope

        self._var_list = None

    @property
    def var_list(self):
        if self._var_list is None:
            self._var_list = tf.trainable_variables('.*' + self.var_scope)

            tf.logging.info('Trainable variables in %s:', self.var_scope)
            for var in self._var_list:
                tf.logging.info('%s', var)

        return self._var_list

    def train_op(self, loss, global_step=None):
        grads = self.compute_grad(loss)
        return self.apply_gradients(grads, global_step=global_step)

    def compute_grad(self, loss):
        """
        Computes gradients of trainable variables with regard to a loss given.

        :param tf.Tensor loss: A `tf.Tensor` of a loss.
        :return: A list of tuples containing gradients and corresponding variables.
        :rtype: list
        """
        return self.optimizer.compute_gradients(loss, var_list=self.var_list)

    def apply_gradients(self, grads_and_vars, global_step=None):
        tf.contrib.training.add_gradients_summaries(grads_and_vars)
        return self.optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    def apply_tower_gradients(self, tower_grads, global_step=None):
        """
        Returns average gradients.

        :param list tower_grads: A list of tuples containing gradients and corresponding variables.
        :return: Applying gradients operation
        :rtype: tf.Operation
        """
        avg_grad = self._average_gradients(tower_grads)
        apply_grad_op = self.apply_gradients(avg_grad, global_step=global_step)
        return apply_grad_op

    @staticmethod
    def _average_gradients(tower_grads):
        if len(tower_grads) == 1:
            return tower_grads[0]

        def filter_no_gradients(grad_and_vars):
            return not any(map(lambda (g, _): g is None, grad_and_vars))

        def reduce_gradients(grad_and_vars):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = map(lambda (g, _): g, grad_and_vars)

            # Get average gradients
            grads = tf.stack(grads)
            grad = tf.reduce_mean(grads, axis=0)

            # Get chief variable
            var = grad_and_vars[0][1]
            return grad, var

        return map(reduce_gradients, filter(filter_no_gradients, zip(*tower_grads)))

    @staticmethod
    def dying_decay(starter_learning_rate, train_iters, decay_iters, decay_steps):
        global_step = tf.train.get_or_create_global_step()

        start_decay = train_iters - decay_iters
        decay_rate = 1. - float(decay_steps) / float(decay_iters)
        learning_rate = tf.where(
            tf.greater_equal(global_step, start_decay),
            x=tf.train.exponential_decay(starter_learning_rate,
                                         global_step - start_decay,
                                         decay_steps,
                                         decay_rate),
            y=starter_learning_rate
        )

        return learning_rate

    @staticmethod
    def step_decay(starter_learning_rate, train_iters, decay_steps, decay_rate):
        global_step = tf.train.get_or_create_global_step()

        num_decay = train_iters / decay_steps

        boundaries = [decay_steps * (i + 1) for i in range(num_decay)]
        values = [starter_learning_rate * decay_rate ** i for i in range(num_decay + 1)]

        learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)

        return learning_rate
