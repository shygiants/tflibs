""" Package utils"""

import importlib
import pkgutil


def import_module(package, module):
    return importlib.import_module('{}.{}'.format(package, module))


def list_modules(package):
    return [name for _, name, ispkg in pkgutil.iter_modules([package]) if not ispkg]
