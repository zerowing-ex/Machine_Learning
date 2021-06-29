from abc import ABCMeta, abstractmethod
import numpy as np


class Optimizer(metaclass=ABCMeta):
    @abstractmethod
    def minimize(self, w, g):
        pass


class SGD(Optimizer):
    def __init__(self,
                 learning_rate=0.01,
                 momentum=0.0,
                 nesterov=False,
                 name="SGD",
                 **kwargs
                 ):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov
        self.name = name
        self.kwargs = kwargs
        self.velocity = None

    def minimize(self, w, g):
        if self.momentum <= 0:
            w -= self.learning_rate * g
        elif not self.nesterov:
            velocity = self.momentum * self.velocity - self.learning_rate * g
            w += velocity
        else:
            velocity = self.momentum * self.velocity - self.learning_rate * g
            w += self.momentum * velocity - self.learning_rate * g


optimizers_dict: dict = {
    'sgd': SGD,
}


def get(optimizer_name):
    optimizer_name = optimizer_name.lower()
    return optimizers_dict[optimizer_name]
