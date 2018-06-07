""" Session """

import tensorflow as tf


def generator(fetches):
    # TODO: Multi-thread
    with tf.Session() as sess:
        while True:
            try:
                yield sess.run(fetches)
            except tf.errors.OutOfRangeError:
                tf.logging.info('tflibs.session.generator: Done')
                break
