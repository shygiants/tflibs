"""
    Writer
"""
import os
from typing import Callable, Iterable
from itertools import cycle

import tensorflow as tf
from tqdm import tqdm

from tflibs.dataset.enums import Split
from tflibs.io.utils import build_tfrecord_basepath
from tflibs.utils.multiprocessing import async_map


class Writer:
    def __init__(self, dataset_dir: str, filename: str, split: Split = None):
        self._dataset_dir = dataset_dir
        self._filename = filename
        self._split = split

        os.makedirs(dataset_dir, exist_ok=True)

    @property
    def dataset_dir(self):
        return self._dataset_dir

    @property
    def filename(self):
        return self._filename

    @property
    def split(self):
        return self._split

    @property
    def tfrecord_basepath(self):
        return build_tfrecord_basepath(self.dataset_dir, self.filename, split=self.split)

    def write(self, collection: Iterable, map_fn: Callable = None, num_shards: int = 32, num_parallel_calls: int = 32):
        try:
            num_examples = len(collection)
        except TypeError:
            num_examples = None

        if num_shards == 1:
            writers = [tf.io.TFRecordWriter(self.tfrecord_basepath)]
        else:
            fname, ext = os.path.splitext(self.tfrecord_basepath)

            postfix_format = '{shard_idx:03d}-of-{num_shards:03d}'
            tfrecord_paths = ['_'.join([fname, postfix_format.format(shard_idx=shard_idx, num_shards=num_shards)]) + ext
                              for shard_idx in range(num_shards)]
            writers = list(map(tf.io.TFRecordWriter, tfrecord_paths))

        if callable(map_fn):
            collection = async_map(map_fn, collection, num_parallel_calls=num_parallel_calls)

        writer_iterator = cycle(writers)

        for i, example in tqdm(enumerate(collection), total=num_examples):
            writer = next(writer_iterator)
            writer.write(example)

        for writer in writers:
            writer.close()


if __name__ == '__main__':
    from tflibs.dataset.dataset_spec import DatasetSpec
    from tflibs.dataset.feature_spec import IDSpec
    from tflibs.io.reader import Reader

    tf.enable_eager_execution()

    ###########
    # Writing #
    ###########
    # collection = map(map_fn, range(1000))
    collection = range(100)

    dataset_spec = DatasetSpec()


    def map_fn(elem):
        id_spec = dataset_spec.feature_specs['_id']  # type: IDSpec
        return dataset_spec.build_example_proto(_id=id_spec.create_with_string(str(elem)))


    dataset_dir = '/Users/kakao/Desktop/tflibs-test'
    filename = 'test.tfrecord'

    writer = Writer(dataset_dir, filename)
    writer.write(collection, map_fn=map_fn)

    ############
    # Training #
    ############

    """
    A task decides Metrics
    """

    """ Dataset """
    reader = Reader(dataset_dir, filename)
    dataset = reader.read()

    dataset = dataset_spec.parse(dataset)

    # TODO: Build input pipeline

    """ Hyperparameters """

    # TODO: Select a model
    # tf.keras.Model

    # TODO: Select a loss
    # Callable

    # TODO: Select an optimizer
    # tf.keras.optimizers.Optimizer

        # TODO: Select an lr schedule if needed
        # tf.keras.optimizers.schedules.LearningRateSchedule

    """ Training step """
    # TODO: Calculate loss

    # TODO: Calculate gradients

    # TODO: Apply gradients

    """ Prepare running """
    # TODO: Summary writer

    # TODO: Distributed training

    """ Training loop """
    # TODO: Restore a model

    # TODO: Run training step

    # TODO: Logging
    # TODO: Evaluation

    # TODO: Save a model

    for elem in dataset:
        print(elem)
