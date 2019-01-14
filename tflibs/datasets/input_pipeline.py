"""
    Input pipeline
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def build_input_fn(dataset,
                   batch_size,
                   map_fn=None,
                   global_step=None,
                   num_elements=None,
                   consider_num_elements=True,
                   shuffle_and_repeat=True,
                   num_parallel_batches=1,
                   shuffle_buffer_size=50,
                   prefetch_buffer_size=20):
    if consider_num_elements:
        num_elements = num_elements or _count_dataset(dataset)
        tf.logging.info('Number of elements in dataset {}: {}'.format(dataset, num_elements))

        if global_step:
            dataset = dataset.skip(((global_step - 1) * batch_size) % num_elements)

    if shuffle_and_repeat:
        dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(batch_size * shuffle_buffer_size))

    if map_fn is not None:
        dataset = dataset.apply(tf.data.experimental.map_and_batch(map_fn, batch_size, drop_remainder=True,
                                                                   num_parallel_batches=num_parallel_batches))
    else:
        dataset = dataset.batch(batch_size, drop_remainder=True)

    dataset = dataset.prefetch(buffer_size=batch_size * prefetch_buffer_size)

    def input_fn():
        iterator = dataset.make_one_shot_iterator()

        return iterator.get_next()

    return input_fn


def _count_dataset(dataset: tf.data.Dataset):
    iterator = dataset.make_one_shot_iterator()
    elem = iterator.get_next()

    with tf.Session() as sess:
        counts = 0
        while True:
            try:
                sess.run(elem)
                counts += 1
            except tf.errors.OutOfRangeError:
                break

    return counts
