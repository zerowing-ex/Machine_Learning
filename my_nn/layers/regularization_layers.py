from base_layers import Layer


class Dropout(Layer):
    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        super().__init__()
        self.__rate = rate

    def forward(self):
        return self.outputs

    def backward(self):
        return

    def get_weights_grads(self):
        return list()

    def __call__(self, last_l, *args, **kwargs):
        return
