import numpy as np
from my_nn.layers.base_layers import Layer
import my_nn.initializers as initializers


class BatchNormalization(Layer):
    def __init__(self, axis=0,
                 momentum=0.99,
                 epsilon=0.001,
                 center=True,
                 scale=True,
                 beta_initializer="zeros",
                 gamma_initializer="ones",
                 moving_mean_initializer="zeros",
                 moving_variance_initializer="ones",
                 # beta_regularizer=None,
                 # gamma_regularizer=None,
                 # beta_constraint=None,
                 # gamma_constraint=None,
                 **kwargs):
        super().__init__()
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale

        self.beta_initializer = None
        self.gamma_initializer = None
        self.moving_mean_initializer = None
        self.moving_variance_initializer = None

        if beta_initializer is not None:
            self.beta_initializer = initializers.get(beta_initializer)()
        if gamma_initializer is not None:
            self.gamma_initializer = initializers.get(gamma_initializer)()
        if moving_mean_initializer is not None:
            self.moving_mean_initializer = initializers.get(moving_mean_initializer)()
        if moving_variance_initializer is not None:
            self.moving_variance_initializer = initializers.get(moving_variance_initializer)()

        self.beta = None
        self.gamma = None
        self.sample_mean = None
        self.sample_variance = None
        self.moving_mean = 0
        self.moving_variance = 1

        self.z = None
        # self.x_minus_mean = None
        self.var_plus_eps = None

        # preserve derivatives
        self.d_beta = None
        self.d_gamma = None

    def forward(self):
        x = self.last_layer[0].outputs

        if self.trainable:
            # normalize to the standard normal distribution
            self.sample_mean = x.mean(axis=self.axis, keepdims=True)
            self.sample_variance = x.var(axis=self.axis, keepdims=True)

            self.z = (x - self.sample_mean) / np.sqrt(self.sample_variance + self.epsilon)
            self.outputs = self.gamma * self.z + self.beta

            # update moving average
            self.moving_mean = self.momentum * self.moving_mean + (1 - self.momentum) * self.sample_mean
            self.moving_variance = self.momentum * self.moving_variance + (1 - self.momentum) * self.sample_variance
        else:
            self.z = (x - self.moving_mean) / np.sqrt(self.moving_variance + self.epsilon)
            self.outputs = self.gamma * self.z + self.beta
        return

    def backward(self):
        # extract variables
        x = self.last_layer[0].outputs
        N = x.shape[0]

        # weights' derivatives
        self.d_beta = np.sum(self.grads, axis=self.axis)
        self.d_gamma = np.sum(self.z * self.grads, axis=self.axis)

        # backward propagation
        d_grads_dz = self.grads * self.gamma

        new_grads = N * d_grads_dz - np.sum(d_grads_dz, axis=self.axis)
        new_grads -= self.z * np.sum(self.z * d_grads_dz, axis=self.axis)
        new_grads /= N * np.sqrt(self.sample_variance + self.epsilon)

        # give grads to the last layers
        for tmp_last_layer in self.last_layer:
            tmp_last_layer.grads = new_grads
        return

    def get_weights_grads(self):
        return list([[self.beta, self.d_beta], [self.gamma, self.d_gamma]])

    def __call__(self, last_l, *args, **kwargs):
        last_l.next_layer.append(self)
        self.last_layer.append(last_l)

        if len(self.last_layer) != 1:
            raise ValueError("Detect incorrect connection between this layer to the last layer")

        if type(last_l.outputs_shape) == int:
            self.outputs_shape = last_l.outputs_shape
        elif len(last_l.outputs_shape) > 2:
            self.outputs_shape = last_l.outputs_shape[1::]
        elif len(last_l.outputs_shape) == 2:
            self.outputs_shape = last_l.outputs_shape[1]
        else:
            raise ValueError(f'Got unsupported output shape {last_l.outputs_shape}')

        self.beta = self.beta_initializer(shape=self.outputs_shape)
        self.gamma = self.gamma_initializer(shape=self.outputs_shape)
        self.moving_mean = self.moving_mean_initializer(shape=self.outputs_shape)
        self.moving_variance = self.moving_variance_initializer(shape=self.outputs_shape)
        return self


class BatchRenormalization:
    def __init__(self):
        return
