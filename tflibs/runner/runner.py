"""
    Runner
"""
import os
import argparse

import tensorflow as tf
import yaml

from tflibs.utils import Attributes


class Runner:
    def __init__(self, use_strategy=True, use_global_step=True, use_summary=True, ignore_unknown=False):
        self._use_strategy = use_strategy
        self._use_global_step = use_global_step
        self._use_summary = use_summary
        self._ignore_unknown = ignore_unknown

        self._argparser = argparse.ArgumentParser()

        self.argparser.add_argument('--config-path',
                                    type=str,
                                    help='')
        self.argparser.add_argument('--verbosity',
                                    choices=[
                                        'DEBUG',
                                        'ERROR',
                                        'FATAL',
                                        'INFO',
                                        'WARN',
                                    ],
                                    default='INFO',
                                    help='Set logging verbosity')

    @property
    def use_strategy(self):
        return self._use_strategy

    @property
    def use_global_step(self):
        return self._use_global_step

    @property
    def use_summary(self):
        return self._use_summary

    @property
    def ignore_unknown(self):
        return self._ignore_unknown

    @property
    def argparser(self):
        return self._argparser

    def run(self, main, yaml_doc=None):
        parse_args, unknown = self.argparser.parse_known_args()

        # Set python level verbosity
        tf.compat.v1.logging.set_verbosity(parse_args.verbosity)
        # Set C++ Graph Execution level verbosity
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(
            tf.compat.v1.logging.__dict__[parse_args.verbosity] / 10)
        del parse_args.verbosity

        if unknown and not self.ignore_unknown:
            raise ValueError('Unknown arguments: {}'.format(unknown))

        #######
        # RUN #
        #######
        if yaml_doc is None:
            with open(parse_args.config_path) as f:
                yaml_doc = f.read()

        args = {}

        def run():
            if self.use_global_step:
                global_step = tf.Variable(1, dtype=tf.int64)
                args.update(global_step=global_step)

            config = yaml.load(yaml_doc, Loader=yaml.Loader)
            # TODO: Log config
            tf.compat.v1.logging.info(yaml_doc)
            config = Attributes(**config)

            if self.use_summary:
                summary_dir = config.job_dir
                train_writer = tf.summary.create_file_writer(summary_dir)
                eval_writer = tf.summary.create_file_writer(os.path.join(summary_dir, 'eval'))
                args.update(train_writer=train_writer, eval_writer=eval_writer)

            main(**args, **vars(config))

        if self.use_strategy:
            strategy = tf.distribute.MirroredStrategy()
            args.update(strategy=strategy)

            with strategy.scope():
                run()

        else:
            run()
