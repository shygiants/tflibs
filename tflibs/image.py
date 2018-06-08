"""
    Image utils
    ~~~~~~~~~~~
"""

import cv2
import numpy as np


def encode(arr):
    """
    Encodes numpy.ndarray into jpeg string

    :param numpy.ndarray arr: numpy.ndarray of an image
    :return: jpeg encoded string
    :rtype: str
    """
    return cv2.imencode('.jpg', arr[:, :, ::-1])[1].tostring()


def decode(string):
    """
    Decodes jpeg string into numpy.ndarray

    :param str string: jpeg encoded string
    :return: numpy.ndarray of a decoded image
    :rtype: numpy.ndarray
    """
    return cv2.imdecode(np.fromstring(string, np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1]
