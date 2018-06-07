""" Dataset """

import os
import threading

import tensorflow as tf
import numpy as np
from tqdm import tqdm

from tflibs.utils import map_dict, flatten_nested_dict
from tflibs.datasets.feature_spec import IDSpec


class BaseDataset:
    def __init__(self, dataset_dir):
        self._dataset_dir = dataset_dir

        if not tf.gfile.Exists(dataset_dir):
            tf.gfile.MakeDirs(dataset_dir)

        self._feature_specs = self._init_feature_specs()
        self._feature_specs.update({
            '_id': IDSpec()
        })

    @property
    def feature_specs(self):
        return self._feature_specs

    @property
    def tfrecord_filename(self):
        raise NotImplementedError

    def _init_feature_specs(self):
        raise NotImplementedError

    @classmethod
    def add_arguments(cls, parser):
        pass

    def write(self, collection, process_fn, num_parallel_calls=16, test_size=None):
        def process_wrapper(coll, thread_idx, test_size):
            # Make tfrecord writer
            fname, ext = os.path.splitext(self.tfrecord_filename)
            fname_pattern = '{fname}{split}{thread_idx:03d}-of-{num_threads:03d}{ext}'
            kwargs = {
                'fname': fname,
                'split': '_{split}_' if test_size is not None else '_',
                'thread_idx': thread_idx,
                'num_threads': num_parallel_calls,
                'ext': ext
            }
            tfrecord_filepattern = os.path.join(self._dataset_dir, fname_pattern.format(**kwargs))

            if test_size is not None:
                train_writer = tf.python_io.TFRecordWriter(tfrecord_filepattern.format(split='train'))
                test_writer = tf.python_io.TFRecordWriter(tfrecord_filepattern.format(split='test'))

                get_writer = lambda i: test_writer if i < test_size else train_writer
            else:
                writer = tf.python_io.TFRecordWriter(os.path.join(self._dataset_dir, self.tfrecord_filename))
                get_writer = lambda _: writer

            for i, elem in tqdm(enumerate(coll), total=len(coll), position=thread_idx):
                # Process
                processed = process_fn(elem, self.feature_specs)
                if processed is None:
                    continue

                # Write
                writer = get_writer(i)

                # Build feature proto
                nested_feature = map_dict(lambda k, v: (k, v.feature_proto(processed[k])), self.feature_specs)
                # Flatten nested dict
                feature = flatten_nested_dict(nested_feature)

                # Build example proto
                example = tf.train.Example(features=tf.train.Features(feature=feature))

                # Write example proto on tfrecord file
                writer.write(example.SerializeToString())

        # Split collection
        spacing = np.linspace(0, len(collection), num_parallel_calls + 1, dtype=np.int)
        ranges = zip(spacing[:-1], spacing[1:])

        for i, rng in enumerate(ranges):
            kwargs = {'coll': collection[rng[0]:rng[1]],
                      'test_size': test_size // num_parallel_calls if test_size is not None else None,
                      'thread_idx': i}

            thread = threading.Thread(target=process_wrapper, kwargs=kwargs)
            thread.start()

    def read(self, split=None):
        def parse(record):
            return map_dict(lambda k, v: (k, v.parse(k, record)), self.feature_specs)

        fname, ext = os.path.splitext(self.tfrecord_filename)
        fname_pattern = '{fname}{split}*{ext}'
        kwargs = {
            'fname': fname,
            'split': '_{split}_'.format(split=split) if split is not None else '_',
            'ext': ext
        }
        tfrecord_filepattern = os.path.join(self._dataset_dir, fname_pattern.format(**kwargs))

        tf.logging.info('TFRecord file pattern: {}'.format(tfrecord_filepattern))
        tf.logging.info('Number of TFRecord files: {}'.format(len(tf.gfile.Glob(tfrecord_filepattern))))

        dataset = tf.data.TFRecordDataset(tf.data.Dataset.list_files(tfrecord_filepattern, shuffle=False))
        dataset = dataset.map(parse)

        return dataset
