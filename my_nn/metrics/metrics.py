from abc import ABCMeta, abstractmethod

import numpy as np


class Metric(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, y_pred, y_true, *args, **kwargs):
        pass


class Accuracy(Metric):
    def __call__(self, y_pred, y_true, *args, **kwargs):
        """
        ========
        Examples
        ========
        >>> y_true = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        >>> y_pred = np.array([[0.05, 0.90, .05], [0.1, 0.8, 0.1], [0.2, 0.7, 0.1]])
        >>> print(Accuracy()(y_pred, y_true))

        >>> y_true = np.array([[0.7], [.01], [1]])
        >>> y_pred = np.array([[1], [0], [0]])
        >>> print(Accuracy()(y_pred, y_true))
        \n
        :param y_pred:
        :param y_true:
        :param args:
        :param kwargs:
        :return:
        """
        if y_pred.shape[-1] != 1:
            tmp_arr = np.argmax(y_pred, axis=-1) == np.argmax(y_true, axis=-1)
        else:
            tmp_arr = np.abs(y_pred - y_true)
            tmp_arr = tmp_arr <= 0.5

        return np.count_nonzero(tmp_arr) / y_pred.shape[0]


metrics_dict: dict = {
    'acc': Accuracy
}


def get(metric_name):
    metric_name = metric_name.lower()
    return metrics_dict[metric_name]
