__all__ = ['get', 'LossFunction', 'BinaryCrossentropy', 'CategoricalCrossentropy', 'SparseCategoricalCrossentropy']

from my_nn.losses.loss import get, LossFunction
from my_nn.losses.probabilistic_losses import BinaryCrossentropy, CategoricalCrossentropy, SparseCategoricalCrossentropy
from my_nn.losses.regression_losses import MeanSquaredError
