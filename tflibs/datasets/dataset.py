"""
    When defining a dataset, a class that inherits `tflibs.datasets.BaseDataset` should be defined.

    The following is an example of a definition.

    >>> import os
    >>> from tflibs.datasets import ImageSpec, LabelSpec
    >>>
    >>> class CatDogDataset(BaseDataset):
    >>>     def __init__(self, dataset_dir, image_size):
    >>>         self._image_size = image_size
    >>>         BaseDataset.__init__(self, os.path.join(dataset_dir, 'cat_dog'))
    >>>
    >>>     @property
    >>>     def tfrecord_filename(self):
    >>>         return 'cat_dog.tfrecord'
    >>>
    >>>     def _init_feature_specs(self):
    >>>         return {
    >>>             'image': ImageSpec([self._image_size, self._image_size, 3]),
    >>>             'label': LabelSpec(3, class_names=['Cat', 'Dog', 'Cookie'])
    >>>         }
    >>>
    >>>     @classmethod
    >>>     def add_arguments(cls, parser):
    >>>         parser.add_argument('--image-size',
    >>>                             type=int,
    >>>                             default=128,
    >>>                             help='The size of output image.')

    When writing TF-record files, create dataset object and call `write()`.

    >>> dataset = CatDogDataset('/tmp/dataset', 64)
    >>>
    >>> images = ['/cat/abc.jpg', '/dog/def.jpg', '/cookie/ghi.jpg']
    >>> labels = ['Cat', 'Dog', 'Cookie']
    >>>
    >>> def process_fn((image_path, label_str), feature_specs):
    >>>     id_string = os.path.splitext(os.path.basename(image_path))[0]
    >>>
    >>>     def build_example(_id, image, label):
    >>>         return {
    >>>             '_id': _id.create_with_string(id_string),
    >>>             'image': image.create_with_path(image_path),
    >>>             'label': label.create_with_label(label_str),
    >>>         }
    >>>
    >>>     return build_example(**feature_specs)
    >>>
    >>> dataset.write(zip(images, labels), process_fn)

    When reading TF-record files, create dataset object and call `read()`.

    >>> dataset = CatDogDataset('/tmp/dataset', 64)
    >>>
    >>> # Returns a `tf.data.Dataset`
    >>> # {
    >>> #   '_id': {
    >>> #       'dtype': tf.string,
    >>> #       'shape': (),
    >>> #   },
    >>> #   'image': {
    >>> #       'dtype': tf.uint8,
    >>> #       'shape': [64, 64, 3],
    >>> #   },
    >>> #   'label': {
    >>> #       'dtype': tf.int64,
    >>> #       'shape': [3],
    >>> #   }
    >>> # }
    >>> tfdataset = dataset.read()


"""

import os
import threading

import tensorflow as tf
import numpy as np
from tqdm import tqdm

from tflibs.utils import map_dict, flatten_nested_dict
from tflibs.datasets.feature_spec import IDSpec


class BaseDataset:
    """
    A base class for defining a dataset

    :param str dataset_dir: A directory where tfrecord files are stored
    """
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
        """
        It should return the name of TF-record file.

        This should be implemented when defining a dataset.

        :return: TF-record filename
        :rtype: str
        """
        raise NotImplementedError

    def _init_feature_specs(self):
        raise NotImplementedError

    @classmethod
    def add_arguments(cls, parser):
        """
        Adds arguments.

        Called when `tflibs.runner.DatasetInitializer <./Initializers.html#tflibs.runner.initializer.DatasetInitializer>`_ creates a dataset object.

        :param argparse.ArgumentParser parser: Argument parser used to add arguments
        """
        pass

    def write(self, collection, process_fn, num_parallel_calls=16, test_size=None):
        """
        Writes examples on tfrecord files

        :param list collection:
        :param function process_fn:
        :param int num_parallel_calls:
        :param int test_size:
        """
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
                writer = tf.python_io.TFRecordWriter(os.path.join(self._dataset_dir, tfrecord_filepattern))
                get_writer = lambda _: writer

            for i, elem in tqdm(enumerate(coll), total=len(coll), position=thread_idx):
                # Process
                processed = process_fn(elem, self.feature_specs, None if test_size is None else i < test_size)
                if processed is None:
                    continue

                # Write
                writer = get_writer(i)

                if not isinstance(processed, list):
                    processed = [processed]

                for processed_e in processed:
                    # Build feature proto
                    nested_feature = map_dict(lambda k, v: (k, v.feature_proto(processed_e[k])), self.feature_specs)
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
        """
        Reads tfrecord and makes it tf.data.Dataset

        :param split: Split name (train or test)
        :return: A dataset
        """
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