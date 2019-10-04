"""
    See the guides: `Dataset <./Dataset.html>`_
"""
from functools import reduce
from operator import mul
from typing import Dict

import tensorflow as tf
import numpy as np
from enum import Enum

from tflibs.utils import CachedProperty, imdecode, imencode


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

    def __init__(self, required=True):
        self._required = required

    @property
    def required(self):
        return self._required

    @property
    def feature_proto_specs(self):
        """
        A property for specifying inner encoding spec of the feature

        :return: The dict containing shape and dtype info
        :rtype: dict
        """
        raise NotImplementedError

    def build_feature_protos(self, value_dict: dict) -> dict:
        """
        Returns a dict of `tf.train.Feature` proto corresponding to `feature_proto_spec`

        :param dict value_dict: A dict containing values of `feature_proto_spec`
        :return: A dict of `tf.train.Feature`
        :rtype: dict
        """

        def build_feature_proto(feature_proto_spec, value):
            dtype = feature_proto_spec['dtype']

            if isinstance(value, np.ndarray):
                value = value.tolist()

            if dtype == tf.string:
                return FeatureSpec._bytes_feature(value)
            elif dtype.is_floating:
                return FeatureSpec._float_feature(value)
            elif dtype.is_integer:
                return FeatureSpec._int64_feature(value)

        return {k: build_feature_proto(feature_proto_spec, value_dict[k])
                for k, feature_proto_spec in self.feature_proto_specs.items() if k in value_dict}

    @CachedProperty
    def features(self) -> Dict[str, tf.io.FixedLenFeature]:
        def build_feature(feature_proto_spec):
            if 'default' in feature_proto_spec:
                default_value = feature_proto_spec['default']
            elif 'required' in feature_proto_spec and not feature_proto_spec['required']:
                size = reduce(mul, feature_proto_spec['shape'], 1)
                dtype = feature_proto_spec['dtype']  # type: tf.DType

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

            return tf.io.FixedLenFeature(feature_proto_spec['shape'], feature_proto_spec['dtype'],
                                         default_value=default_value)

        return {k: build_feature(feature_spec) for k, feature_spec in self.feature_proto_specs.items()}

    def build_tensor(self, parsed: Dict[str, tf.Tensor]):
        raise NotImplementedError


class ScalarSpec(FeatureSpec):
    def __init__(self, dtype: tf.DType):
        super(ScalarSpec, self).__init__()
        self._dtype = dtype

    @property
    def tfdtype(self):
        return self._dtype

    @property
    def pydtype(self):
        tfdtype = self._dtype

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

    def create_with_value(self, value) -> dict:
        if not isinstance(value, self.pydtype):
            value = self.pydtype(value)

        return self.build_feature_protos({
            'value': value
        })

    def build_tensor(self, parsed: Dict[str, tf.Tensor]):
        return parsed['value']


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

    def create_with_string(self, string: str) -> dict:
        return self.build_feature_protos({
            '_id': string
        })

    def build_tensor(self, parsed: Dict[str, tf.Tensor]):
        return parsed['_id']


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

    def create_with_path(self, path: str) -> dict:
        with open(path, 'r') as f:
            decoded = imdecode(f.read())
            assert decoded.shape[0] == self.image_shape[0] and decoded.shape[1] == self.image_shape[1]

            return self.build_feature_protos({
                'encoded': imencode(decoded)
            })

    def create_with_tensor(self, arr: np.ndarray) -> dict:
        assert arr.shape[0] == self.image_shape[0] and arr.shape[1] == self.image_shape[1]

        return self.build_feature_protos({
            'encoded': imencode(arr)
        })

    def create_with_contents(self, contents: str) -> dict:
        decoded = imdecode(contents)
        assert decoded.shape[0] == self.image_shape[0] and decoded.shape[1] == self.image_shape[1]

        return self.build_feature_protos({
            'encoded': imencode(decoded)
        })

    def build_tensor(self, parsed: Dict[str, tf.Tensor]):
        decoded = tf.image.decode_png(parsed['encoded'], channels=self.channels)
        decoded = tf.reshape(decoded, self.image_shape)

        return decoded


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

    def create_with_path(self, path: str) -> dict:
        with open(path, 'rb') as f:
            return self.build_feature_protos({
                'encoded': f.read()
            })

    def create_with_tensor(self, tensor: np.ndarray) -> dict:
        return self.build_feature_protos({
            'encoded': tfimage.encode(tensor)
        })

    def create_with_contents(self, contents: str) -> dict:
        return self.build_feature_protos({
            'encoded': contents
        })

    def build_tensor(self, parsed: Dict[str, tf.Tensor]):
        decoded = tf.image.decode_image(parsed['encoded'], channels=self.channels)
        decoded.set_shape((None, None, self.channels))

        return decoded


class LabelSpec(FeatureSpec):
    """
    A class for specifying one-hot label spec

    :param int depth: The number of labels
    :param list class_names: A list of `str` which describes each labels
    """

    def __init__(self, depth: int, class_names=None):
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

    def create_with_index(self, index: int) -> dict:
        assert index < self.depth

        return self.build_feature_protos({
            'index': index
        })

    def create_with_label(self, label) -> dict:
        # TODO: Assert bad labels don't exist
        return self.build_feature_protos({
            'index': self._class_names.index(label)
        })

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

    def build_tensor(self, parsed: Dict[str, tf.Tensor]):
        return tf.one_hot(parsed['index'], self.depth)


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

    def create_with_tensor(self, tensor) -> dict:
        # TODO: Assert shape
        return self.build_feature_protos({
            'tensor': tensor
        })

    def create_with_labels(self, labels) -> dict:
        # TODO: Assert bad labels don't exist
        return self.build_feature_protos({
            'tensor': list(map(lambda cls: int(cls in labels), self._class_names))
        })

    def build_tensor(self, parsed: Dict[str, tf.Tensor]):
        return parsed['tensor']


class EnumSpec(FeatureSpec):
    """
    A class for specifying enum.

    :param enum_cls:
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

    def create_with_string(self, string: str) -> dict:
        if string not in [lambda e: e.value, list(self.enum)]:
            raise ValueError('String should be one of members ({})'.format(list(self.enum)))

        return self.build_feature_protos({
            'enum': string,
        })

    def create_with_member(self, member: Enum) -> dict:
        return self.build_feature_protos({
            'enum': member.value,
        })

    def build_tensor(self, parsed: Dict[str, tf.Tensor]):
        return parsed['enum']
