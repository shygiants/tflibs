"""
    Enums for dataset
"""
from enum import Enum


class Split(Enum):
    Train = 'train'
    Valid = 'valid'
    Test = 'test'
