"""
    Utils
"""
import os

from tflibs.dataset.enums import Split


def build_tfrecord_basepath(dataset_dir: str, filename: str, split: Split = None):
    tfrecord_basepath = os.path.join(dataset_dir, filename)

    if split is not None:
        fname, ext = os.path.splitext(tfrecord_basepath)
        fname = '_'.join([fname, split.value])
        tfrecord_basepath = fname + ext

    return tfrecord_basepath
