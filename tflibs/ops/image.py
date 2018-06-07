"""
    Image ops
    ~~~~~~~~~
"""

import tensorflow as tf


def normalize(images):
    images = tf.image.convert_image_dtype(images, tf.float32)
    # [0, 1] --> [-1, 1]
    images -= 0.5
    images *= 2.

    return images


def decode_image(encoded_image, image_shape, image_format=None, conditional=False):
    # TODO: Fuse decode and crop
    if conditional:
        image = tf.cond(tf.equal(image_format, 'raw', name='is_raw'),
                        true_fn=lambda: tf.decode_raw(encoded_image, tf.uint8),
                        false_fn=lambda: tf.image.decode_image(encoded_image),
                        name='decoded_image')
    else:
        image = tf.image.decode_image(encoded_image)
    image = tf.reshape(image, image_shape)

    return image
