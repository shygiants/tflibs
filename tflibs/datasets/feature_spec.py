""" Feature Spec """

import tensorflow as tf
import numpy as np

from tflibs.utils import map_dict
from tflibs import image as tfimage


class FeatureSpec:
    """A class for specifying `Feature` proto.
    """

    @staticmethod
    def _int64_feature(values):
        """Returns a TF-Feature of int64s.

        Args:
          values: A scalar or list of values.

        Returns:
          A TF-Feature.
        """
        if not isinstance(values, (tuple, list)):
            values = [values]
        return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

    @staticmethod
    def _bytes_feature(values):
        """Returns a TF-Feature of bytes.

        Args:
            values: A string.

        Returns:
            A TF-Feature.
        """
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

    @staticmethod
    def _float_feature(values):
        """Returns a TF-Feature of floats.

        Args:
            values: A scalar of list of values.

        Returns:
            A TF-Feature.
        """
        if not isinstance(values, (tuple, list)):
            values = [values]
        return tf.train.Feature(float_list=tf.train.FloatList(value=values))

    def __init__(self, shape):
        self._shape = shape

    @property
    def shape(self):
        return self._shape

    @property
    def feature_proto_spec(self):
        raise NotImplementedError

    def feature_proto(self, value_dict):
        def map_fn(k, v):
            dtype = v['dtype']
            value = value_dict[k]

            if isinstance(value, np.ndarray):
                value = value.tolist()

            if dtype == tf.string:
                return k, FeatureSpec._bytes_feature(value)
            elif dtype == tf.float16 or dtype == tf.float32 or dtype == tf.float64:
                return k, FeatureSpec._float_feature(value)
            elif dtype == tf.int8 or dtype == tf.int16 or dtype == tf.int32 or dtype == tf.int64:
                return k, FeatureSpec._int64_feature(value)

        return map_dict(map_fn, self.feature_proto_spec)

    def parse(self, parent_key, record):
        def parse(k, v):
            return '{}/{}'.format(parent_key, k), tf.FixedLenFeature(v['shape'], v['dtype'])

        features = map_dict(parse, self.feature_proto_spec)
        return tf.parse_single_example(record, features)


class IDSpec(FeatureSpec):
    def __init__(self):
        FeatureSpec.__init__(self, ())

    @property
    def feature_proto_spec(self):
        return {
            '_id': {
                'shape': (),
                'dtype': tf.string,
            }
        }

    def parse(self, parent_key, record):
        parsed = FeatureSpec.parse(self, parent_key, record)

        return parsed['{}/_id'.format(parent_key)]

    def create_with_string(self, string):
        return {
            '_id': string
        }


class ImageSpec(FeatureSpec):
    def __init__(self, image_size):
        image_size = list(image_size)
        FeatureSpec.__init__(self, image_size + [3])

    @property
    def feature_proto_spec(self):
        return {
            'encoded': {
                'shape': (),
                'dtype': tf.string,
            }
        }

    def parse(self, parent_key, record):
        parsed = FeatureSpec.parse(self, parent_key, record)
        decoded = tf.image.decode_image(parsed['{}/{}'.format(parent_key, 'encoded')], channels=3)
        decoded = tf.reshape(decoded, self.shape)

        return decoded

    def create_with_path(self, path):
        # TODO: Assert shape
        with open(path, 'rb') as f:
            return {
                'encoded': f.read()
            }

    def create_with_tensor(self, tensor):
        # TODO: Assert shape
        return {
            'encoded': tfimage.encode(tensor)
        }

    def create_with_contents(self, contents):
        # TODO: Assert shape

        return {
            'encoded': contents
        }


class LabelSpec(FeatureSpec):
    def __init__(self, depth, class_names=None):
        FeatureSpec.__init__(self, [depth])
        self._class_names = class_names

    @property
    def feature_proto_spec(self):
        return {
            'index': {
                'shape': (),
                'dtype': tf.int64
            }
        }

    def parse(self, parent_key, record):
        parsed = FeatureSpec.parse(self, parent_key, record)

        return tf.one_hot(parsed['{}/{}'.format(parent_key, 'index')], self.shape[0])

    def create_with_index(self, index):
        assert index < self.shape[0]

        return {
            'index': index
        }

    @classmethod
    def from_class_names(cls, class_names):
        obj = cls(len(class_names), class_names)

        return obj


class MultiLabelSpec(FeatureSpec):
    def __init__(self, depth, class_names=None):
        FeatureSpec.__init__(self, [depth])
        self._class_names = class_names

    @property
    def feature_proto_spec(self):
        return {
            'tensor': {
                'shape': self.shape,
                'dtype': tf.int64
            }
        }

    @classmethod
    def from_class_names(cls, class_names):
        obj = cls(len(class_names), class_names)

        return obj

    def parse(self, parent_key, record):
        parsed = FeatureSpec.parse(self, parent_key, record)

        return parsed['{}/{}'.format(parent_key, 'tensor')]

    def create_with_tensor(self, tensor):
        # TODO: Assert shape
        return {
            'tensor': tensor
        }

    def create_with_labels(self, labels):
        # TODO: Assert bad labels don't exist
        return {
            'tensor': map(lambda cls: int(cls in labels), self._class_names)
        }
