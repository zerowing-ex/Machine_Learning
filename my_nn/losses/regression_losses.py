from my_nn.losses.loss import LossFunction, losses_dict


class MeanSquaredError(LossFunction):
    def forward(self, y_pred, y_true):
        return

    def backward(self):
        return


losses_dict['mse'] = MeanSquaredError
