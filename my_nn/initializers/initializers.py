from abc import ABCMeta, abstractmethod

import numpy as np


class Initializer(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, shape, dtype=None, *args, **kwargs):
        pass


class RandomNormal(Initializer):
    def __init__(self, mean=0.0, stddev=0.05):
        self.mean = mean
        self.stddev = stddev

    def __call__(self, shape, dtype=None, *args, **kwargs):
        return np.random.normal(loc=self.mean, scale=self.stddev, size=shape)


class RandomUniform(Initializer):
    def __init__(self, minval=-0.05, maxval=0.05):
        self.minval = minval
        self.maxval = maxval

    def __call__(self, shape, dtype=None, *args, **kwargs):
        return np.random.uniform(low=self.minval, high=self.maxval, size=shape)


class Zeros(Initializer):
    def __call__(self, shape, dtype=None, *args, **kwargs):
        return np.zeros(shape)


class Ones(Initializer):
    def __call__(self, shape, dtype=None, *args, **kwargs):
        return np.ones(shape)


initializers_dict: dict = {
    'random_normal': RandomNormal,
    'random_uniform': RandomUniform,
    'zeros': Zeros,
    'ones': Ones,
}


def get(init_name):
    return initializers_dict[init_name]
