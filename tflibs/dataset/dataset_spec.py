"""
    Dataset Spec
"""
from typing import Dict

import tensorflow as tf

from tflibs.dataset.enums import Split
from tflibs.utils import CachedProperty, flatten_nested_dict
from tflibs.dataset.feature_spec import FeatureSpec, IDSpec


class DatasetSpec:

    @CachedProperty
    def feature_specs(self) -> Dict[str, FeatureSpec]:
        return {
            '_id': IDSpec(),
        }

    @CachedProperty
    def train_feature_specs(self) -> dict:
        return self.feature_specs

    @CachedProperty
    def valid_feature_specs(self) -> dict:
        return self.feature_specs

    @CachedProperty
    def test_feature_specs(self) -> dict:
        return self.feature_specs

    def build_example_proto(self, **features) -> tf.train.Example:
        for k, feature_spec in self.feature_specs.items():
            if feature_spec.required and k not in features:
                raise ValueError('`{}` should be provided'.format(k))

        features = flatten_nested_dict(features)

        return tf.train.Example(features=tf.train.Features(feature=features)).SerializeToString()

    def parse(self, dataset: tf.data.TFRecordDataset, split: Split = None) -> tf.data.Dataset:
        # TODO: Consider split
        feature_specs = self.feature_specs  # type: Dict[str, FeatureSpec]
        features = {k: feature_spec.features for k, feature_spec in feature_specs.items()}
        flatten_features = flatten_nested_dict(features)

        def parse_fn(record: tf.Tensor):
            parsed = tf.io.parse_single_example(record, flatten_features)

            unflatten = {key: [] for key in features.keys()}

            # Unflatten
            for flatten_key, tensor in parsed.items():
                key, inner_key = flatten_key.split('/')

                unflatten[key].append((inner_key, tensor))

            return {k: feature_specs[k].build_tensor(dict(tups)) for k, tups in unflatten.items()}

        return dataset.map(parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
