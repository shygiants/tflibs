"""
    Session Run Hooks
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading
import os
import tensorflow as tf
from tflibs.image import encode


class EvaluationRunHook(tf.train.SessionRunHook):
    def __init__(self, estimator, input_fn, eval_steps, summary=True):
        self.estimator = estimator
        self.input_fn = input_fn
        self.eval_steps = eval_steps
        self.summary = summary
        self._lock = threading.Lock()

    def before_run(self, run_context):
        return tf.train.SessionRunArgs({'global_step': tf.train.get_or_create_global_step()})

    def after_run(self,
                  run_context,  # pylint: disable=unused-argument
                  run_values):
        global_step = run_values.results['global_step']
        if (global_step + 1) % self.eval_steps == 0:
            # Start new thread for evaluation
            t = threading.Thread(target=self._run_evaluation)
            t.start()

    def _run_evaluation(self):
        if self._lock.acquire(False):
            try:
                self.estimator.evaluate(self.input_fn, hooks=[
                    EvalSummaryHook(os.path.join(self.estimator.model_dir, 'eval'))
                ] if self.summary else None)
            finally:
                self._lock.release()


class EvalSummaryHook(tf.train.SessionRunHook):
    def __init__(self,
                 summary_dir,
                 summary_op=None):
        self._summary_dir = summary_dir
        self._summary_op = summary_op
        self._summary_writer = None
        self._finished = False

    def begin(self):
        self._summary_op = self._summary_op or tf.summary.merge_all()
        self._summary_writer = tf.summary.FileWriterCache.get(self._summary_dir)

    def before_run(self, run_context):
        return tf.train.SessionRunArgs({
            'summary': self._summary_op,
            'global_step': tf.train.get_or_create_global_step()
        }) if not self._finished and self._summary_op is not None else None

    def after_run(self,
                  run_context,  # pylint: disable=unused-argument
                  run_values):
        if not self._finished:
            self._summary_writer.add_summary(run_values.results['summary'], run_values.results['global_step'])
            self._finished = True


class ImageSaverHook(tf.train.SessionRunHook):
    def __init__(self,
                 images,
                 image_dir):
        self.run_args = dict(
            map(lambda item: (item[0], tf.image.convert_image_dtype(item[1], tf.uint8)), images.items()))
        self.image_dir = image_dir
        self.iter = 0

        if not tf.gfile.Exists(self.image_dir):
            tf.gfile.MakeDirs(self.image_dir)

        tf.logging.info('ImageSaverHook: {}'.format(self.run_args))

    def before_run(self, run_context):
        run_args = dict(self.run_args)
        run_args.update({'global_step': tf.train.get_or_create_global_step()})
        return tf.train.SessionRunArgs(run_args)

    def after_run(self,
                  run_context,  # pylint: disable=unused-argument
                  run_values):
        global_step = run_values.results['global_step']
        for k in self.run_args.keys():
            images = run_values.results[k]
            encoded = encode(images)
            with open(os.path.join(self.image_dir,
                                   '{key}_{gs:07d}_{iter:03d}.jpg'.format(
                                       key=k, gs=global_step, iter=self.iter)), 'wb') as f:
                f.write(encoded)

        self.iter += 1
