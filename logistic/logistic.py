import numpy as np


def sigmoid(z):
    a = 1 / (1 + np.exp(-z))
    return a


class Logistic:
    def __init__(self):
        """
        :return None
        """
        """preserve params
        self.m: 样本数量
        self.w: shape: (self.m)，权重
        self.b: 偏置"""
        self.__m = None
        self.__m_valid = None

        self.__w = None
        self.__b = None

        self.__learning_rate = None

        # preserve inputs and true labels
        self.__x = None
        self.__y = None
        self.__x_valid = None
        self.__y_valid = None

        # preserve outputs
        self.__z = None
        self.__a = None

        # preserve derivatives
        self.__dz = None
        self.__dw = None

        # cost and accuracy
        self.__loss: float = np.inf
        self.__loss_valid: float = np.inf

        self.__acc: float = 0
        self.__acc_valid: float = 0

    def __forward(self):
        # calculate outputs
        # (m, 1) = (m, N) × (N, 1)
        self.__z = np.dot(self.__x, self.__w) + self.__b
        self.__a = sigmoid(self.__z)

        self.__loss_acc()
        return

    def __loss_acc(self):
        # calculate cost
        self.__loss = - np.sum(self.__y * np.log(self.__a) + (1 - self.__y) * np.log(1 - self.__a)) / self.__m

        z = np.dot(self.__x_valid, self.__w) + self.__b
        a = sigmoid(z)
        self.__loss_valid = - np.sum(self.__y_valid * np.log(a) + (1 - self.__y_valid) * np.log(1 - a)) / self.__m_valid

        # calculate accuracy
        tmp_arr = np.abs(self.__a - self.__y)
        self.__acc = np.count_nonzero(tmp_arr <= 0.5) / self.__m
        tmp_arr = np.abs(a - self.__y_valid)
        self.__acc_valid = np.count_nonzero(tmp_arr <= 0.5) / self.__m_valid

    def __backward(self):
        self.__dz = self.__a - self.__y  # (m, 1)
        self.__dw = np.dot(self.__x.T, self.__dz) / self.__m  # (N, 1) = (N, m) × (m, 1)
        self.db = np.sum(self.__dw) / self.__m
        return

    def __optimize(self):
        """
        利用传统的梯度下降法\n
        :return: None
        """
        self.__w -= self.__learning_rate * self.__dw
        self.__b -= self.__learning_rate * self.db
        return

    def predict(self, x: np.ndarray):
        z = np.dot(x, self.__w) + self.__b
        a = sigmoid(z)
        return a

    # def evaluate(self, x: np.ndarray,
    #              y: np.ndarray,
    #              set_type: str = 'train'):
    #     accuracy = 1 - np.mean(np.abs(self.predict(x) - y))
    #     print("Accuracy on {} set: {:.4f}".format(set_type, accuracy))

    def fit(self, x_train: np.ndarray,
            y_train: np.ndarray,
            x_valid: np.ndarray,
            y_valid: np.ndarray,
            epochs: int = 200,
            verbose: int = 1,
            learning_rate: float = 1e-3):
        """
        :param x_train: shape: (samples, None)，训练集样本
        :param y_train: shape: (samples, 1)，训练集样本所对应的标签：0/1
        :param x_valid: shape: (samples, None)，验证集样本
        :param y_valid: shape: (samples, 1)，验证集样本所对应的标签：0/1
        :param epochs: 设置epoch
        :param verbose: 设置大于1的数以开启提示
        :param learning_rate: 学习率，默认为0.001
        """
        if x_train.ndim != 2:
            raise ValueError(
                f"The shape of the training set is illegal, got {x_train.shape}, expected [samples, features]")
        if y_train.ndim != 2 or y_train.shape[-1] != 1:
            raise ValueError(
                f"The shape of the training label is illegal, got {y_train.shape}, expected [samples, 1]")
        if x_valid.ndim != 2:
            raise ValueError(
                f"The shape of the validation set is illegal, got {x_valid.shape}, expected [samples, features]")
        if y_valid.ndim != 2 or y_valid.shape[-1] != 1:
            raise ValueError(
                f"The shape of the validation label is illegal, got {y_valid.shape}, expected [samples, 1]")

        self.__m = x_train.shape[0]
        self.__m_valid = x_valid.shape[0]
        # (N, 1)
        self.__w = np.zeros((x_train.shape[1], 1))
        self.__b = 0

        self.__learning_rate = learning_rate

        # preserve inputs and true labels
        self.__x = x_train
        self.__y = y_train

        self.__x_valid = x_valid
        self.__y_valid = y_valid

        # 先前向传播，然后后向传播
        for epoch in range(1, epochs + 1):
            self.__forward()
            self.__backward()
            self.__optimize()

            if verbose > 0:
                print("Epoch {}/{} - loss: {:.4f} - acc: {:.4f} - val_loss: {:.4f} "
                      "- val_acc: {:.4f}".format(epoch, epochs, self.__loss, self.__acc, self.__loss_valid,
                                                 self.__acc_valid))
        return


"""Run Sample"""
if __name__ == "__main__":
    from dataset.catvnoncat import load_dataset

    x_train, y_train, x_valid, y_valid = load_dataset()

    x_train = x_train.reshape(x_train.shape[0], -1)
    x_valid = x_valid.reshape(x_valid.shape[0], -1)
    x_train = (x_train / 255).astype(np.float64)
    x_valid = (x_valid / 255).astype(np.float64)

    y_train = np.expand_dims(y_train, axis=-1)
    y_valid = np.expand_dims(y_valid, axis=-1)

    print(y_train.shape)
    # model = Logistic()
    # model.fit(x_train, y_train, x_valid, y_valid, epochs=20000)
