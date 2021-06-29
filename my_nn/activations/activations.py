from abc import ABCMeta, abstractmethod

import numpy as np


class Activation(metaclass=ABCMeta):
    def __init__(self):
        self.a = None
        self.da = None

    @abstractmethod
    def forward(self, z):
        pass

    @abstractmethod
    def backward(self):
        pass


class Identity(Activation):
    def forward(self, z):
        return z

    def backward(self):
        return 1


class Softmax(Activation):
    def __init__(self, axis=-1):
        super().__init__()
        self.__axis = axis
        return

    def forward(self, z):
        if z.ndim <= 1:
            raise ValueError(f"The input dimension of softmax activation is illegal, got {z.ndim}")
        sums = np.sum(np.exp(z), axis=self.__axis, keepdims=True)
        self.a = np.exp(z) / sums
        return self.a

    def backward(self):
        return self.a * (1 - self.a)


class Sigmoid(Activation):
    def forward(self, z):
        self.a = 1 / (1 + np.exp(-z))
        return self.a

    def backward(self):
        return self.a * (1 - self.a)


class Tanh(Activation):
    def forward(self, z):
        self.a = (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
        return self.a

    def backward(self):
        return 1 - self.a * self.a


class ReLU(Activation):
    def forward(self, z):
        self.a = np.maximum(z, 0)
        self.da = z > 0
        return self.a

    def backward(self):
        return self.da


class LeakyReLU(Activation):
    def __init__(self, alpha: float = 0.1):
        super().__init__()
        self.__alpha = alpha

    def forward(self, z):
        self.a = np.maximum(z, self.__alpha * z)
        self.da = z > self.__alpha * z
        self.da[not self.da] = self.__alpha
        return self.a

    def backward(self):
        return self.da


activations_dict: dict = {
    'identity': Identity,
    'softmax': Softmax,
    'sigmoid': Sigmoid,
    'tanh': Tanh,
    'relu': ReLU,
    'leakyReLU': LeakyReLU
}


def get(act_name):
    return activations_dict[act_name]


# if __name__ == "__main__":
#     x: np.ndarray = np.array([
#         [4, 3, 4],
#         [1, 2, 3]
#     ])
#     act = Softmax()
#     print(act.forward(x))
#     print(act.backward())
