from queue import Queue
import sys
import time
import numpy as np

import my_nn.losses
import my_nn.metrics
import my_nn.optimizers
from my_nn.layers import *


class Model:
    def __init__(self, inputs: Layer, outputs: Layer):
        # preserve input layer and output layer
        self.__inputs = inputs
        self.__outputs = outputs

        # optimizer/loss function/metrics
        self.__optimizer = my_nn.optimizers.SGD()
        self.__loss = my_nn.losses.CategoricalCrossentropy()
        self.__metrics = my_nn.metrics.Accuracy()
        # forward propagation and backward propagation
        self.__compute_sequence: list = list()

        # batch size
        self.__batch_size = None

        # preserve inputs and true labels
        self.__x_train = None
        self.__y_train = None
        self.__x_valid = None
        self.__y_valid = None

        # cost and accuracy
        self.__loss_train: float = np.inf
        self.__loss_valid: float = np.inf

        self.__acc_train: float = 0
        self.__acc_valid: float = 0

        # fit history
        self.__history: dict = dict()
        return

    def __construct_compute_sequence(self):
        # delete all items
        self.__compute_sequence.clear()
        # maintain a queue
        q: Queue = Queue()
        q.put(self.__inputs)
        while not q.empty():
            tmp_layer = q.get()
            self.__compute_sequence.append(tmp_layer)
            if tmp_layer == self.__outputs:
                return
            next_layer_list: list = tmp_layer.next_layer
            for next_layer in next_layer_list:
                q.put(next_layer)

    def __forward(self):
        # get valid loss and valid acc
        self.__compute_sequence[0].inputs = self.__x_valid
        self.__compute_sequence[0].outputs = self.__x_valid
        for tmp_layer in self.__compute_sequence[1::]:
            tmp_layer.training = False
            tmp_layer.forward()

        self.__loss_valid = self.__loss.forward(self.__outputs.outputs, self.__y_valid)
        self.__acc_valid = self.__metrics(self.__outputs.outputs, self.__y_valid)

        # get train loss and train acc
        self.__compute_sequence[0].inputs = self.__x_train
        self.__compute_sequence[0].outputs = self.__x_train
        for tmp_layer in self.__compute_sequence[1::]:
            tmp_layer.training = True
            tmp_layer.forward()

        self.__loss_train = self.__loss.forward(self.__outputs.outputs, self.__y_train)
        self.__acc_train = self.__metrics(self.__outputs.outputs, self.__y_train)
        return

    def __backward_optimize(self):
        # get loss function gradients
        loss_fun_grads = self.__loss.backward()
        # set output layer gradients manually
        length: int = len(self.__compute_sequence)
        self.__compute_sequence[-1].grads = loss_fun_grads
        for i in range(length - 1):
            tmp_layer = self.__compute_sequence[length - 1 - i]
            tmp_layer.backward()
            # optimizing
            for weight_pair in tmp_layer.get_weights_grads():
                self.__optimizer.minimize(weight_pair[0], weight_pair[1])
        return

    def compile(self,
                optimizer="sgd",
                loss=None,
                metrics=None,
                # loss_weights=None,
                # weighted_metrics=None,
                # run_eagerly=None,
                # steps_per_execution=None,
                # **kwargs
                ):
        if type(optimizer) == str:
            self.__optimizer = my_nn.optimizers.get(optimizer)()
        else:
            self.__optimizer = optimizer

        if type(loss) == str:
            self.__loss = my_nn.losses.get(loss)()
        else:
            self.__loss = loss

        if type(metrics) == str:
            self.__metrics = my_nn.metrics.get(metrics)()
        else:
            self.__metrics = metrics
        return

    def fit(self,
            x=None,
            y=None,
            batch_size=None,
            epochs=200,
            verbose=1,
            # verbose="auto",
            # callbacks=None,
            # validation_split=0.0,
            validation_data=None,
            # shuffle=True,
            # class_weight=None,
            # sample_weight=None,
            # initial_epoch=0,
            # steps_per_epoch=None,
            # validation_steps=None,
            # validation_batch_size=None,
            # validation_freq=1,
            # max_queue_size=10,
            # workers=1,
            # use_multiprocessing=False
            ):
        self.__x_valid, self.__y_valid = validation_data[0], validation_data[1]
        self.__batch_size = batch_size
        self.__construct_compute_sequence()

        samples: int = x.shape[0]
        steps_per_epoch: int = samples // batch_size

        for epoch in range(1, epochs + 1):
            if verbose > 0:
                print(f"Epoch {epoch}/{epochs}")

            for step in range(1, steps_per_epoch + 1):
                start, stop = (step - 1) * batch_size, min(step * batch_size, samples)
                # start timer
                start_time: float = time.process_time()

                # get tmp training set
                self.__x_train = x[start: stop, ...]
                self.__y_train = y[start: stop, ...]

                self.__forward()
                self.__backward_optimize()

                # end timer
                end_time: float = time.process_time()
                step_cost_time: float = end_time - start_time

                if verbose > 0:
                    numbers: int = 30 * step // steps_per_epoch

                    progress_bar: str = numbers * '='
                    if numbers == 30:
                        progress_bar += '='
                    else:
                        progress_bar += '>'
                    progress_bar += (30 - numbers - 1) * ' '

                    show_str: str = "{}/{} [{}] - {}s {}ms/step - loss: {:.4f} - accuracy: {:.4f} - val_loss: {:.4f} - val_accuracy: {:.4f}". \
                        format(step, steps_per_epoch, progress_bar, int(step_cost_time * steps_per_epoch),
                               int(1000 * step_cost_time),
                               self.__loss_train, self.__acc_train, self.__loss_valid, self.__acc_valid)

                    if step == steps_per_epoch:
                        sys.stdout.write("\r{0}\n".format(show_str))
                        sys.stdout.flush()
                    else:
                        sys.stdout.write("\r{0}".format(show_str))
                        sys.stdout.flush()

        return self.__history


"""Run Sample"""
if __name__ == "__main__":
    # from dataset.catvnoncat import load_dataset
    # from my_nn.utils import to_one_hot
    #
    # x_train, y_train, x_valid, y_valid = load_dataset()
    #
    # x_train = x_train.reshape(x_train.shape[0], -1)
    # x_valid = x_valid.reshape(x_valid.shape[0], -1)
    # x_train = (x_train / 255).astype(np.float64)
    # x_valid = (x_valid / 255).astype(np.float64)
    #
    # y_train = np.expand_dims(y_train, axis=-1)
    # y_valid = np.expand_dims(y_valid, axis=-1)
    # y_train = to_one_hot(y_train)
    # y_valid = to_one_hot(y_valid)

    from dataset.MNIST import load_dataset
    from my_nn.utils import to_one_hot

    x_train, y_train, x_valid, y_valid = load_dataset()

    x_train = x_train.reshape(x_train.shape[0], -1)
    x_valid = x_valid.reshape(x_valid.shape[0], -1)
    x_train = (x_train / 255).astype(np.float64)
    x_valid = (x_valid / 255).astype(np.float64)
    y_train = to_one_hot(y_train)
    y_valid = to_one_hot(y_valid)

    nn_input = Input(shape=(x_train.shape[1]))
    x = Dense(1024, activation='relu')(nn_input)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    nn_output = Dense(10, activation='softmax')(x)

    model = Model(inputs=nn_input, outputs=nn_output)
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics='acc')
    model.fit(x_train, y_train, validation_data=(x_valid, y_valid), batch_size=8192)
