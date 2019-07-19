"""
    See the guides: `Dataset <./Dataset.html>`_
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import reduce
from operator import mul

import tensorflow as tf
import numpy as np
from enum import Enum

from tflibs.utils import map_dict
from tflibs import image as tfimage


class FeatureSpec:
    """A class for specifying feature specs of a `tflibs.datasets.BaseDataset <./Dataset.html>`_.
    """

    @staticmethod
    def _int64_feature(values):
        """
        Returns a TF-Feature of int64s.

        :param int or list values: A int values.
        :return: A TF-Feature proto.
        :rtype: tf.train.Feature
        """
        if not isinstance(values, (tuple, list)):
            values = [values]
        return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

    @staticmethod
    def _bytes_feature(values):
        """
        Returns a TF-Feature of bytes.

        :param str values: A string.
        :return: A TF-Feature proto.
        :rtype: tf.train.Feature
        """
        if not isinstance(values, bytes):
            values = bytes(values, 'utf-8')

        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

    @staticmethod
    def _float_feature(values):
        """
        Returns a TF-Feature of floats.

        :param float or list values: A scalar of list of values.
        :return: A TF-Feature proto.
        :rtype: tf.train.Feature
        """
        if not isinstance(values, (tuple, list)):
            values = [values]
        return tf.train.Feature(float_list=tf.train.FloatList(value=values))

    @property
    def feature_proto_specs(self):
        """
        A property for specifying inner encoding spec of the feature

        :return: The dict containing shape and dtype info
        :rtype: dict
        """
        raise NotImplementedError

    def feature_proto(self, value_dict):
        """
        Returns a dict of `tf.train.Feature` proto corresponding to `feature_proto_spec`

        :param dict value_dict: A dict containing values of `feature_proto_spec`
        :return: A dict of `tf.train.Feature`
        :rtype: dict
        """

        def make_feature(feature_proto_spec, value):
            dtype = feature_proto_spec['dtype']

            if isinstance(value, np.ndarray):
                value = value.tolist()

            if dtype == tf.string:
                return FeatureSpec._bytes_feature(value)
            elif dtype == tf.float16 or dtype == tf.float32 or dtype == tf.float64:
                return FeatureSpec._float_feature(value)
            elif dtype == tf.int8 or dtype == tf.int16 or dtype == tf.int32 or dtype == tf.int64:
                return FeatureSpec._int64_feature(value)

        return {k: make_feature(feature_proto_spec, value_dict[k])
                for k, feature_proto_spec in self.feature_proto_specs.items() if k in value_dict}

    def parse(self, parent_key, record):
        """
        Parse TF-record and returns dict of `tf.Tensor`

        :param str parent_key: The key of the feature
        :param tf.Tensor record: String tensor of TF-record
        :return: A dict of tensors as specified at `feature_proto_spec`
        :rtype: dict
        """

        def make_feature(feature_proto):
            if 'default' in feature_proto:
                default_value = feature_proto['default']
            elif 'required' in feature_proto and not feature_proto['required']:
                size = reduce(mul, feature_proto['shape'], 1)
                dtype = feature_proto['dtype']  # type: tf.DType

                if dtype.is_floating:
                    value = -1.
                elif dtype.is_integer:
                    value = -1
                elif dtype == tf.string:
                    value = ''
                else:
                    raise ValueError('')

                if size == 0:
                    default_value = value
                else:
                    default_value = [value] * size
            else:
                default_value = None

            return tf.FixedLenFeature(feature_proto['shape'], feature_proto['dtype'],
                                      default_value=default_value)

        features = {
            '{}/{}'.format(parent_key, k): make_feature(feature_spec)
            for k, feature_spec in self.feature_proto_specs.items()
        }

        parsed = tf.parse_single_example(record, features)

        return {k.split('/')[-1]: v for k, v in parsed.items()}


class ScalarSpec(FeatureSpec):
    def __init__(self, dtype):
        super(ScalarSpec, self).__init__(())
        self._dtype = dtype

    @property
    def tfdtype(self):
        return self._dtype

    @property
    def pydtype(self):
        tfdtype = self._dtype  # type: tf.DType

        if tfdtype.is_integer:
            return int
        elif tfdtype.is_floating:
            return float
        elif tfdtype == tf.string:
            return str
        else:
            raise ValueError('Invalid dtype')

    @property
    def feature_proto_specs(self):
        return {
            'value': {
                'shape': (),
                'dtype': self.tfdtype,
            }
        }

    def create_with_value(self, value):
        if not isinstance(value, self.pydtype):
            value = self.pydtype(value)

        return {
            'value': value
        }


class IDSpec(FeatureSpec):
    """A class for specifying unique ID.
    """

    @property
    def feature_proto_specs(self):
        """
        A property for specifying inner encoding spec of the feature

        :return: The dict containing shape and dtype info
        :rtype: dict
        """
        return {
            '_id': {
                'shape': (),
                'dtype': tf.string,
            }
        }

    def parse(self, parent_key, record):
        """
        Parse TF-record and returns `tf.Tensor`

        :param str parent_key: The key of the feature
        :param tf.Tensor record: String tensor of TF-record
        :return: A scalar tensor containing an id
        :rtype: tf.Tensor
        """
        parsed = FeatureSpec.parse(self, parent_key, record)

        return parsed['_id']

    def create_with_string(self, string):
        return {
            '_id': string
        }


class ImageSpec(FeatureSpec):
    """
    A class for specifying image spec

    :param list|tuple image_shape: The sizes of images
    """

    def __init__(self, image_shape):
        self._image_shape = tuple(image_shape)

    @property
    def image_shape(self):
        return self._image_shape

    @property
    def channels(self):
        return self.image_shape[-1]

    @property
    def feature_proto_specs(self):
        """
        A property for specifying inner encoding spec of the feature

        :return: The dict containing shape and dtype info
        :rtype: dict
        """
        return {
            'encoded': {
                'shape': (),
                'dtype': tf.string,
            }
        }

    def parse(self, parent_key, record):
        """
        Parse TF-record and returns `tf.Tensor`

        :param str parent_key: The key of the feature
        :param tf.Tensor record: String tensor of TF-record
        :return: A 3-D tensor containing an image
        :rtype: tf.Tensor
        """
        parsed = FeatureSpec.parse(self, parent_key, record)
        decoded = tf.image.decode_image(parsed['encoded'], channels=self.channels)
        decoded = tf.reshape(decoded, self.image_shape)

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


class VarImageSpec(FeatureSpec):
    """
    A class for specifying image spec

    :param list|tuple image_size: The sizes of images
    """

    def __init__(self, channels):
        self._channels = channels

    @property
    def feature_proto_specs(self):
        """
        A property for specifying inner encoding spec of the feature

        :return: The dict containing shape and dtype info
        :rtype: dict
        """
        return {
            'encoded': {
                'shape': (),
                'dtype': tf.string,
            }
        }

    @property
    def channels(self):
        return self._channels

    def parse(self, parent_key, record):
        """
        Parse TF-record and returns `tf.Tensor`

        :param str parent_key: The key of the feature
        :param tf.Tensor record: String tensor of TF-record
        :return: A 3-D tensor containing an image
        :rtype: tf.Tensor
        """
        parsed = FeatureSpec.parse(self, parent_key, record)
        decoded = tf.image.decode_image(parsed['encoded'], channels=self.channels)
        decoded.set_shape((None, None, self.channels))

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
    """
    A class for specifying one-hot label spec

    :param int depth: The number of labels
    :param list class_names: A list of `str` which describes each labels
    """

    def __init__(self, depth, class_names=None):
        self._depth = depth
        self._class_names = class_names

    @property
    def depth(self):
        return self._depth

    @property
    def feature_proto_specs(self):
        """
        A property for specifying inner encoding spec of the feature

        :return: The dict containing shape and dtype info
        :rtype: dict
        """
        return {
            'index': {
                'shape': (),
                'dtype': tf.int64
            }
        }

    def parse(self, parent_key, record):
        """
        Parse TF-record and returns `tf.Tensor`

        :param str parent_key: The key of the feature
        :param tf.Tensor record: String tensor of TF-record
        :return: An 1-D tensor containing label
        :rtype: tf.Tensor
        """
        parsed = FeatureSpec.parse(self, parent_key, record)

        return tf.one_hot(parsed['index'], self.depth)

    def create_with_index(self, index):
        assert index < self.depth

        return {
            'index': index
        }

    def create_with_label(self, label):
        # TODO: Assert bad labels don't exist
        return {
            'index': self._class_names.index(label)
        }

    @classmethod
    def from_class_names(cls, class_names):
        """
        Create `LabelSpec` object from a list of names specifying each classes

        :param list class_names: A list of names specifying each classes
        :return: `LabelSpec` object
        :rtype: LabelSpec
        """
        obj = cls(len(class_names), class_names)

        return obj


class MultiLabelSpec(FeatureSpec):
    """
    A class for specifying multi-label spec

    :param int depth: The number of labels
    :param list class_names: A list of `str` which describes each labels
    """

    def __init__(self, depth, class_names=None):
        self._depth = depth
        self._class_names = class_names

    @property
    def depth(self):
        return self._depth

    @property
    def feature_proto_specs(self):
        """
        A property for specifying inner encoding spec of the feature

        :return: The dict containing shape and dtype info
        :rtype: dict
        """
        return {
            'tensor': {
                'shape': (self.depth,),
                'dtype': tf.int64
            }
        }

    @classmethod
    def from_class_names(cls, class_names):
        """
        Create `MultiLabelSpec` object from a list of names specifying each classes

        :param list class_names: A list of names specifying each classes
        :return: `MultiLabelSpec` object
        :rtype: MultiLabelSpec
        """
        obj = cls(len(class_names), class_names)

        return obj

    def parse(self, parent_key, record):
        """
        Parse TF-record and returns `tf.Tensor`

        :param str parent_key: The key of the feature
        :param tf.Tensor record: String tensor of TF-record
        :return: A 1-D tensor containing label
        :rtype: tf.Tensor
        """
        parsed = FeatureSpec.parse(self, parent_key, record)

        return parsed['tensor']

    def create_with_tensor(self, tensor):
        # TODO: Assert shape
        return {
            'tensor': tensor
        }

    def create_with_labels(self, labels):
        # TODO: Assert bad labels don't exist
        return {
            'tensor': list(map(lambda cls: int(cls in labels), self._class_names))
        }


class EnumSpec(FeatureSpec):
    """A class for specifying enum.
    """

    def __init__(self, enum_cls):
        self.enum = enum_cls

    @property
    def feature_proto_specs(self):
        """
        A property for specifying inner encoding spec of the feature

        :return: The dict containing shape and dtype info
        :rtype: dict
        """
        return {
            'enum': {
                'shape': (),
                'dtype': tf.string,
            }
        }

    def parse(self, parent_key, record):
        """
        Parse TF-record and returns `tf.Tensor`

        :param str parent_key: The key of the feature
        :param tf.Tensor record: String tensor of TF-record
        :return: A scalar tensor containing an id
        :rtype: tf.Tensor
        """
        parsed = FeatureSpec.parse(self, parent_key, record)

        return parsed['enum']

    def create_with_string(self, string: str):
        if string not in [lambda e: e.value, list(self.enum)]:
            raise ValueError('String should be one of members ({})'.format(list(self.enum)))

        return {
            'enum': string,
        }

    def create_with_member(self, member: Enum):
        return {
            'enum': member.value,
        }
