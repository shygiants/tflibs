"""
    Model
    ~~~~~
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import tensorflow as tf

from tflibs.utils import Attributes, device_setter, image_summary


class _TensorDescriptor:
    def __init__(self, fn=None, device=None, summary=None):
        self._device = device
        self._summary = summary

        self.fn = fn

    def __get__(self, instance, owner):
        with tf.device(instance.device_setter(self.device)):
            val = self.fn(instance)

            if instance.model_idx == 0 and self.summary is not None:
                if val.shape.ndims == 0:
                    tf.summary.scalar(self.summary, val)
                else:
                    tf.summary.histogram(self.summary, val)

                tf.logging.info('Summary {} is defined'.format(self.summary))

        setattr(instance, self.name, val)
        return val

    def __call__(self, fn):
        self.fn = fn

        return self

    @property
    def device(self):
        return self._device

    @property
    def summary(self):
        return self._summary

    @property
    def fn(self):
        return self._fn

    @fn.setter
    def fn(self, new_fn):
        self._fn = new_fn
        if hasattr(new_fn, '__name__'):
            self._name = new_fn.__name__

    @property
    def name(self):
        return self._name


class _ImageDescriptor(_TensorDescriptor):
    def __init__(self, fn=None, device=None, summary=None):
        _TensorDescriptor.__init__(self, fn=fn, device=device, summary=None)

        self._img_summary = summary

    def __get__(self, instance, owner):
        val = _TensorDescriptor.__get__(self, instance, owner)

        if instance.model_idx == 0 and self.img_summary is not None:
            image_summary(self.img_summary, val)
            tf.logging.info('Summary {} is defined at `Images`'.format(self.img_summary))

        return val

    @property
    def img_summary(self):
        return self._img_summary


class _LossDescriptor(_TensorDescriptor):
    def __init__(self, fn=None, device=None, summary=None):
        _TensorDescriptor.__init__(self, fn=fn, device=device, summary=None)

        self._loss_summary = summary

    def __get__(self, instance, owner):
        val = _TensorDescriptor.__get__(self, instance, owner)

        if val not in tf.losses.get_losses():
            val = tf.identity(val, name='{}/value'.format(self.loss_summary))
            tf.losses.add_loss(val)

        return val

    @property
    def loss_summary(self):
        return self._loss_summary


class Model:
    tensor = _TensorDescriptor
    image = _ImageDescriptor
    loss = _LossDescriptor

    def __init__(self, features: dict, labels=None, model_idx=0, model_parallelism=True, device=None, **hparams):
        self._features = features
        self._labels = labels
        self._model_idx = model_idx
        self._model_parallelism = model_parallelism
        self._device = device

        self._hparams = Attributes(**hparams)
        self._networks = Attributes()

    @property
    def features(self):
        return self._features

    @property
    def labels(self):
        return self._labels

    @property
    def model_idx(self):
        return self._model_idx

    @property
    def model_parallelism(self):
        return self._model_parallelism

    @property
    def device(self):
        return self._device

    @property
    def hparams(self):
        return self._hparams

    @property
    def networks(self):
        return self._networks

    @staticmethod
    def model_fn(features, labels, mode, params):
        raise NotImplementedError

    @classmethod
    def num_devices(cls):
        raise NotImplementedError

    @classmethod
    def add_model_args(cls, argparser: argparse.ArgumentParser, parse_args: argparse.Namespace):
        pass

    @classmethod
    def add_train_args(cls, argparser: argparse.ArgumentParser, parse_args: argparse.Namespace):
        pass

    @classmethod
    def add_eval_args(cls, argparser: argparse.ArgumentParser, parse_args: argparse.Namespace):
        pass

    def summary_loss(self):
        losses = tf.losses.get_losses()
        tf.logging.info('Defining summaries of losses')
        for loss in losses:
            lname = loss.name  # type: str
            tokens = lname.split('/')
            while True:
                token = tokens.pop()
                token = token.rstrip(':0')
                if token != 'value':
                    tf.summary.scalar(token, loss, family='Losses')
                    tf.logging.info('Summary {} is defined at `Losses`'.format(token))
                    break

    def device_setter(self, device_id: int):
        if device_id is None:
            return None
        gpu_id = self.device or (
            self.num_devices() * self.model_idx + device_id if self.model_parallelism else self.model_idx)
        return device_setter('/gpu:{}'.format(gpu_id))


class Network:
    def __init__(self, **hparams):
        self._hparams = Attributes(**hparams)

    @property
    def hparams(self):
        return self._hparams

    def __call__(self, *args, **kwargs):
        raise NotImplementedError
