""" Initializers for Runner """

import argparse

import tensorflow as tf

from tflibs.utils import list_modules, import_module, log_parse_args


class BaseInitializer:
    def add_arguments(self, argparser):
        raise NotImplementedError

    def handle(self, parse_args, unknown):
        raise NotImplementedError


class DatasetInitializer(BaseInitializer):
    def __init__(self, dataset_pkg='datasets.dataset'):
        self._dataset_pkg = dataset_pkg

    def _list_datasets(self):
        return list_modules(self._dataset_pkg.replace('.', '/'))

    def _dataset_factory(self, name):
        return import_module(self._dataset_pkg, name).export

    def add_arguments(self, argparser):
        argparser.add_argument('--dataset-dir',
                               required=True,
                               type=str,
                               help='The directory where the dataset files are stored.')
        argparser.add_argument('--dataset-name',
                               type=str,
                               help='The name of a dataset.',
                               choices=self._list_datasets())

    def handle(self, parse_args, unknown):
        dataset_name = parse_args.dataset_name
        dataset_dir = parse_args.dataset_dir

        del parse_args.dataset_name
        del parse_args.dataset_dir

        dataset_cls = self._dataset_factory(dataset_name)

        # Parse dataset-specific arguments
        parser = argparse.ArgumentParser()
        dataset_cls.add_arguments(parser)
        parse_args, unknown = parser.parse_known_args(unknown)
        log_parse_args(parse_args)

        dataset = dataset_cls(dataset_dir, **vars(parse_args))

        return {'dataset': dataset}, unknown


class ModelInitializer(BaseInitializer):
    def __init__(self, model_pkg='models'):
        self._model_pkg = model_pkg

    def _model_factory(self, name):
        return import_module(self._model_pkg, name).export

    def _list_models(self):
        return list_modules(self._model_pkg)

    def add_arguments(self, argparser):
        argparser.add_argument('--model-name',
                               type=str,
                               help='The name of the model to use.',
                               choices=self._list_models())

    def handle(self, parse_args, unknown):
        model_name = parse_args.model_name

        del parse_args.model_name

        model_cls = self._model_factory(model_name)

        return {'model_cls': model_cls}, unknown


class TrainInitializer(ModelInitializer):
    def add_arguments(self, argparser):
        ModelInitializer.add_arguments(self, argparser)

        argparser.add_argument('--save-steps',
                               type=int,
                               default=1000,
                               help='The number of steps to save the model.')
        argparser.add_argument('--keep-checkpoint-max',
                               type=int,
                               default=100,
                               help='The maximum number of checkpoint file to keep.')
        argparser.add_argument('--log-steps',
                               type=int,
                               default=10,
                               help='The number of steps to log the progress.')
        argparser.add_argument('--random-seed',
                               type=int,
                               default=1,
                               help='Random seed')
        argparser.add_argument('--train-iters',
                               type=int,
                               default=200000,
                               help='Maximum number of training iterations to perform.')

    def handle(self, parse_args, unknown):
        handled_args, unknown = ModelInitializer.handle(self, parse_args, unknown)

        model_cls = handled_args['model_cls']

        # Parse model-specific arguments
        parser = argparse.ArgumentParser()
        model_cls.add_model_args(parser, parse_args)
        model_args, unknown = parser.parse_known_args(unknown)
        log_parse_args(model_args)

        # Parse model-specific train arguments
        parser = argparse.ArgumentParser()
        model_cls.add_train_args(parser, parse_args)
        train_args, unknown = parser.parse_known_args(unknown)
        log_parse_args(train_args)

        # Parse model-specific eval arguments
        parser = argparse.ArgumentParser()
        model_cls.add_eval_args(parser, parse_args)
        eval_args, unknown = parser.parse_known_args(unknown)
        log_parse_args(eval_args)

        train_args = vars(train_args)
        train_args.update({'train_iters': parse_args.train_iters})

        model_params = {
            'model_args': vars(model_args),
            'train_args': train_args,
            'eval_args': vars(eval_args),
        }

        save_steps = parse_args.save_steps
        log_steps = parse_args.log_steps,
        keep_checkpoint_max = parse_args.keep_checkpoint_max
        random_seed = parse_args.random_seed

        del parse_args.save_steps
        del parse_args.log_steps,
        del parse_args.keep_checkpoint_max
        del parse_args.random_seed

        session_config = tf.ConfigProto(log_device_placement=False,
                                        allow_soft_placement=False)

        run_config = tf.estimator.RunConfig().replace(session_config=session_config,
                                                      save_summary_steps=save_steps,
                                                      save_checkpoints_steps=save_steps,
                                                      log_step_count_steps=log_steps,
                                                      keep_checkpoint_max=keep_checkpoint_max,
                                                      tf_random_seed=random_seed)

        estimator = tf.estimator.Estimator(
            model_cls.model_fn,
            model_dir=parse_args.job_dir,
            config=run_config,
            params=model_params)

        return {'estimator': estimator}, unknown
