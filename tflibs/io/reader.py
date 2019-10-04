"""
    Reader
"""
import os

import tensorflow as tf

from tflibs.dataset.enums import Split
from tflibs.io.utils import build_tfrecord_basepath


class Reader:
    def __init__(self, dataset_dir: str, filename: str, split: Split = None):
        self._dataset_dir = dataset_dir
        self._filename = filename
        self._split = split

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

    def read(self, num_parallel_reads: int = 32) -> tf.data.TFRecordDataset:
        fname, ext = os.path.splitext(self.tfrecord_basepath)
        filenames = sorted(tf.io.gfile.glob('*'.join([fname, ext])))
        dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=num_parallel_reads)

        return dataset
