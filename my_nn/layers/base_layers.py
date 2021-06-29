from abc import ABCMeta, abstractmethod

import numpy as np

import my_nn.activations as activations
import my_nn.initializers as initializers


class Layer(metaclass=ABCMeta):
    def __init__(self):
        # preserve reference to last and next layer
        self.last_layer = list()
        self.next_layer = list()

        # preserve outputs and grads
        # outputs from tmp layer
        self.outputs = None
        self.outputs_shape = None
        # grads from next layer
        self.grads = None
        # signature if training
        self.trainable: bool = True

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def backward(self):
        pass

    @abstractmethod
    def get_weights_grads(self):
        pass

    @abstractmethod
    def __call__(self, last_l, *args, **kwargs):
        pass


class Input(Layer):
    def __init__(self, shape):
        super().__init__()
        # self.__shape = np.array(shape)
        self.inputs = None
        self.outputs = None
        self.outputs_shape = shape

    def forward(self):
        self.outputs = self.inputs
        return self.outputs

    def backward(self):
        return

    def get_weights_grads(self):
        return list()

    def __call__(self, last_l, *args, **kwargs):
        return


class Dense(Layer):
    def __init__(self, units: int,
                 activation=None,
                 use_bias: bool = True,
                 kernel_initializer="random_normal",
                 bias_initializer="zeros",
                 # kernel_regularizer=None,
                 # bias_regularizer=None,
                 # activity_regularizer=None,
                 # kernel_constraint=None,
                 # bias_constraint=None,
                 name=None):
        super().__init__()
        # params
        self.__units: int = units
        self.outputs_shape = units

        self.__use_bias = use_bias

        self.__activation = activations.Identity()
        self.kernel_initializer = None
        self.bias_initializer = None

        if activation is not None:
            self.__activation = activations.get(activation)()

        if kernel_initializer is not None:
            self.kernel_initializer = initializers.get(kernel_initializer)()

        if use_bias and bias_initializer is not None:
            self.bias_initializer = initializers.get(bias_initializer)()

        # preserve outputs and grads
        # inputs from this layer
        self.this_layer_inputs = None
        # outputs from tmp layer
        self.outputs = None
        # grads from next layer
        self.grads = None

        # params
        self.kernel = None
        self.bias = None

        # preserve outputs
        self.z = None
        self.activity = None

        # preserve derivatives
        self.dz = None
        self.d_kernel = None
        self.d_bias = None
        return

    def forward(self):
        last_layer = self.last_layer[0]
        # forward propagation
        # use bias: z = x·w + b
        # use bias: z = x·w
        if self.__use_bias:
            self.z = np.dot(last_layer.outputs, self.kernel) + self.bias
        else:
            self.z = np.dot(last_layer.outputs, self.kernel)

        self.activity = self.__activation.forward(self.z)

        self.outputs = self.activity
        return

    def backward(self):
        last_layer = self.last_layer[0]

        # get this layer input from last layer
        self.this_layer_inputs = last_layer.outputs

        '''
        ==============================================
        如果下一层为空，则认为是输出层，必须要人为地对梯度进行赋值
        ==============================================
        '''

        # backward propagation
        self.dz = self.__activation.backward() * self.grads

        # weights' derivatives
        self.d_kernel = np.dot(self.this_layer_inputs.T, self.dz) / self.this_layer_inputs.shape[0]
        self.d_bias = np.sum(self.dz, axis=0, keepdims=True) / self.this_layer_inputs.shape[0]

        new_grads = np.dot(self.dz, self.kernel.T)
        # give grads to the last layers
        self.last_layer[0].grads = new_grads
        return

    def get_weights_grads(self):
        result: list = list([(self.kernel, self.d_kernel)])
        if self.__use_bias:
            result.append((self.bias, self.d_bias))
        return result

    def __call__(self, last_l, *args, **kwargs):
        last_l.next_layer.append(self)
        self.last_layer.append(last_l)

        if len(self.last_layer) != 1:
            raise ValueError("Detect incorrect connection between this layer to the last layer")

        # init kernel and bias
        last_layer = self.last_layer[0]
        self.kernel = self.kernel_initializer(shape=[last_layer.outputs_shape, self.__units])
        if self.__use_bias:
            self.bias = self.bias_initializer(shape=[1, self.__units])
        return self
