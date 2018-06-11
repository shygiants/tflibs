"""
    Input pipeline
"""

import tensorflow as tf


def build_input_fn(dataset, batch_size, map_fn=None, global_step=None, num_elements=None, shuffle_and_repeat=True):
    num_elements = num_elements or _count_dataset(dataset)
    tf.logging.info('Number of elements in dataset {}: {}'.format(dataset, num_elements))

    if global_step:
        dataset = dataset.skip(((global_step - 1) * batch_size) % num_elements)

    if shuffle_and_repeat:
        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(batch_size * 50))

    if map_fn is not None:
        dataset = dataset.apply(tf.contrib.data.map_and_batch(
            map_func=map_fn, batch_size=batch_size, drop_remainder=True))
    else:
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))

    dataset = dataset.prefetch(buffer_size=batch_size * 20)

    def input_fn():
        iterator = dataset.make_one_shot_iterator()

        return iterator.get_next()

    return input_fn


def _count_dataset(dataset):
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
