from abc import ABCMeta, abstractmethod


class LossFunction(metaclass=ABCMeta):
    def __init__(self):
        self.y_pred = None
        self.y_true = None

    @abstractmethod
    def forward(self, y_pred, y_true):
        pass

    @abstractmethod
    def backward(self):
        pass

losses_dict: dict = dict()

def get(loss_name):
    return losses_dict[loss_name]
