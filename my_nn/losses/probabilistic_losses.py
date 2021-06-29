from my_nn.losses.loss import LossFunction, losses_dict
import numpy as np


class BinaryCrossentropy(LossFunction):
    def forward(self, y_pred, y_true):
        self.y_pred, self.y_true = y_pred, y_true

        loss = - np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) / y_pred.shape[0]
        return loss

    def backward(self):
        grad = (self.y_pred - self.y_true) / self.y_pred / (1 - self.y_pred)
        return grad


class CategoricalCrossentropy(LossFunction):
    def forward(self, y_pred, y_true):
        self.y_pred, self.y_true = y_pred, y_true

        if y_pred.shape != y_true.shape:
            raise ValueError(
                f"The shape of y_pred and y_true is not equal, namely one-hot form, y_pred {y_pred.shape}, y_true{y_true.shape}")
        if y_pred.ndim != 2 or y_pred.shape[-1] < 2:
            raise ValueError(f"The shape of y_pred and y_true is illegal, got {y_pred.shape}, expected (?, classes)")
        loss = - np.sum((y_true * np.log(y_pred))) / y_pred.shape[0]
        return loss

    def backward(self):
        grad = self.y_pred - self.y_true
        return grad


class SparseCategoricalCrossentropy(LossFunction):
    def forward(self, y_pred, y_true):
        self.y_pred, self.y_true = y_pred, y_true

        return

    def backward(self):
        return


losses_dict['binary_crossentropy'] = BinaryCrossentropy
losses_dict['categorical_crossentropy'] = CategoricalCrossentropy
losses_dict['sparse_categorical_crossentropy'] = SparseCategoricalCrossentropy

# if __name__ == "__main__":
#     y_true = np.array([[0, 1, 0], [0, 0, 1]])
#     y_pred = np.array([[0.05, 0.90, .05], [0.1, 0.8, 0.1]])
#     loss = CategoricalCrossentropy().forward(y_pred, y_true)
#     print(loss)
