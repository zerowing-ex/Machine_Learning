from kernels import *


class SVM:
    """
    Reference:
    https://www.cnblogs.com/pursued-deer/p/7892342.html
    """

    def __init__(self):
        # 样本数量和样本维度
        self.__m, self.__n = None, None
        self.__m_valid = None

        # parameters
        self.__kernel = None

        self.__kernel_dict: dict = {
            'linear': linear, 'Linear': linear,
            'poly': poly, 'Poly': poly,
            'gauss': gauss, 'Gauss': gauss,
            'laplace': laplace, 'Laplace': laplace,
            'Sigmoid': Sigmoid, 'sigmoid': Sigmoid
        }

        self.__b: float = 0.0
        # temp predictions/gradients
        self.__g = None
        # error between predictions and true label
        self.__E = None
        # SMO算法对偶因子
        self.__alpha = None

        # 松弛变量ζ
        self.__Zeta: float = 1.0

        # preserve inputs and true labels
        self.__x = None
        self.__y = None

        self.__x_valid = None
        self.__y_valid = None

        # cost and accuracy
        self.__acc: float = 0
        self.__acc_valid: float = 0

    # g(x)预测值，输入xi（X[i]）
    # g(x) = \sum_{j=1}^N {\alpha_j*y_j*K(x_j,x)+b}
    def __update_E(self):
        tmp_arr: np.ndarray = np.zeros_like(self.__y)
        for i in range(self.__m):
            tmp_arr[i] = self.__b + np.sum(self.__alpha * self.__y * self.__kernel(self.__x[i], self.__x))

        self.__E = tmp_arr - self.__y

    def __KKT_condition(self, i) -> bool:
        y_g = self.__g[i] * self.__y[i]
        if self.__alpha[i] == 0:
            return y_g >= 1
        elif 0 < self.__alpha[i] < self.__Zeta:
            return y_g == 1
        else:
            return y_g <= 1

    def __init_alpha(self):
        # 外层循环首先遍历所有满足0<a<Zeta的样本点，检验是否满足KKT
        indexes = [i for i in range(self.__m) if 0 < self.__alpha[i] < self.__Zeta]
        # 否则遍历整个训练集
        if len(indexes) == 0:
            indexes = np.linspace(start=0, stop=self.__m, num=self.__m, endpoint=False)

        # 外层循环选择满足0<alpha_i<Zeta，且不满足KKT的样本点。如果不存在遍历剩下训练集
        for i in indexes:
            if self.__KKT_condition(i):
                continue
            # 内层循环，|E1-E2|最大化
            E1 = self.__E[i]
            # 如果E1是+，选择最小的E_i作为E2；如果E1是负的，选择最大的E_i作为E2

            if E1 >= 0:
                j = int(np.argmin(self.__E))
            else:
                j = int(np.argmax(self.__E))
            return i, j

    def __main_procedure(self):
        index_1, index_2 = self.__init_alpha()

        # 边界,计算阈值b和差值E_i
        if self.__y[index_1] == self.__y[index_2]:
            # L = max(0, alpha_2 + alpha_1 -C)
            # H = min(C, alpha_2 + alpha_1)
            L = max(0, self.__alpha[index_1] + self.__alpha[index_2] - self.__Zeta)
            H = min(self.__Zeta, self.__alpha[index_1] + self.__alpha[index_2])
        else:
            # L = max(0, alpha_2 - alpha_1)
            # H = min(C, alpha_2 + alpha_1+C)
            L = max(0, self.__alpha[index_2] - self.__alpha[index_1])
            H = min(self.__Zeta, self.__Zeta + self.__alpha[index_2] - self.__alpha[index_1])

        E1 = self.__E[index_1]
        E2 = self.__E[index_2]
        # eta=K11+K22-2K12= ||phi(x_1) - phi(x_2)||^2
        eta = self.__kernel(self.__x[index_1], self.__x[index_1]) + self.__kernel(self.__x[index_2], self.__x[
            index_2]) - 2 * self.__kernel(self.__x[index_1], self.__x[index_2])
        if eta <= 0:
            return
        # 更新约束方向的解
        alpha_2 = self.__alpha[index_2] + self.__y[index_2] * (E1 - E2) / eta  # 此处有修改，根据书上应该是E1 - E2，书上130-131页
        if alpha_2 < L:
            alpha_2 = L
        elif alpha_2 > H:
            alpha_2 = H

        alpha_1 = self.__alpha[index_1] + self.__y[index_1] * self.__y[index_2] * (self.__alpha[index_2] - alpha_2)

        b1_new = -E1 - self.__y[index_1] * self.__kernel(self.__x[index_1], self.__x[index_1]) * (
                alpha_1 - self.__alpha[index_1]) - self.__y[index_2] * self.__kernel(self.__x[index_2],
                                                                                     self.__x[index_1]) * (
                         alpha_2 - self.__alpha[index_2]) + self.__b
        b2_new = -E2 - self.__y[index_1] * self.__kernel(self.__x[index_1], self.__x[index_2]) * (
                alpha_1 - self.__alpha[index_1]) - self.__y[index_2] * self.__kernel(self.__x[index_2],
                                                                                     self.__x[index_2]) * (
                         alpha_2 - self.__alpha[index_2]) + self.__b

        if 0 < alpha_1 < self.__Zeta:
            b_new = b1_new
        elif 0 < alpha_2 < self.__Zeta:
            b_new = b2_new
        else:
            # 选择中点
            b_new = (b1_new + b2_new) / 2

        # 更新参数
        self.__alpha[index_1] = alpha_1
        self.__alpha[index_2] = alpha_2
        self.__b = b_new

        r = self.__b + np.sum(self.__alpha * self.__y * self.__kernel(self.__x[index_1], self.__x))
        self.__E[index_1] = r - self.__y[index_1]
        r = self.__b + np.sum(self.__alpha * self.__y * self.__kernel(self.__x[index_2], self.__x))
        self.__E[index_2] = r - self.__y[index_2]

    def predict(self, x: np.ndarray, inter_use: bool = False):
        predictions = self.__b + np.sum(self.__alpha * self.__x * self.__kernel(x, self.__x))
        predictions[predictions > 0] = 1
        if inter_use:
            predictions[predictions <= 0] = -1
        else:
            predictions[predictions <= 0] = 0
        return predictions

    def __loss_acc(self):
        # calculate accuracy
        predictions = self.predict(self.__x)
        tmp_arr = np.abs(predictions - self.__y)
        self.__acc = np.count_nonzero(tmp_arr <= 1e-3) / self.__m
        predictions = self.predict(self.__x_valid)
        tmp_arr = np.abs(predictions - self.__y_valid)
        self.__acc_valid = np.count_nonzero(tmp_arr <= 1e-3) / self.__m_valid

    def fit(self, x_train: np.ndarray,
            y_train: np.ndarray,
            x_valid: np.ndarray,
            y_valid: np.ndarray,
            kernel='linear',
            epochs: int = 200,
            verbose: int = 1):
        """
        :param x_train: shape: (samples, None)，训练集样本
        :param y_train: shape: (samples, 1)，训练集样本所对应的标签：0/1
        :param x_valid: shape: (samples, None)，验证集样本
        :param y_valid: shape: (samples, 1)，验证集样本所对应的标签：0/1
        :param kernel: 指定使用的核函数
        :param epochs: 设置epoch
        :param verbose: 设置大于1的数以开启提示
        :return:
        """
        if x_train.ndim != 2:
            raise ValueError(f"The shape of the training set is illegal, got {x_train.shape}, expected [samples, features]")
        if y_train.ndim != 2 or y_train.shape[-1] != 1:
            raise ValueError(f"The shape of the training label is illegal, got {y_train.shape}, expected [samples, 1]")
        if x_valid.ndim != 2:
            raise ValueError(f"The shape of the validation set is illegal, got {x_valid.shape}, expected [samples, features]")
        if y_valid.ndim != 2 or y_valid.shape[-1] != 1:
            raise ValueError(f"The shape of the validation label is illegal, got {y_valid.shape}, expected [samples, 1]")
        self.__m, self.__n = x_train.shape[0], x_train.shape[1]
        self.__m_valid = x_valid.shape[0]

        # preserve inputs and true labels
        self.__x = x_train
        self.__y = y_train

        self.__x_valid = x_valid
        self.__y_valid = y_valid

        # change positive and negative label numbers
        # 1 - 1        0 - -1
        self.__y[self.__y == 0] = -1
        self.__y_valid[self.__y_valid == 0] = -1

        # init params
        if kernel not in self.__kernel_dict.keys():
            raise ValueError(f"The kernel type is not supported, got {kernel}, expected 'linear, poly, gauss, laplace, "
                             f"Sigmoid, Linear, Poly, Gauss, Laplace, sigmoid'")
        self.__kernel = self.__kernel_dict[kernel]
        self.__alpha = np.ones(self.__m)

        for epoch in range(1, epochs + 1):
            self.__main_procedure()
            self.__loss_acc()
            if verbose > 0:
                print("Epoch {}/{} - acc: {:.4f} - val_acc: {:.4f}".format(epoch, epochs, self.__acc, self.__acc_valid))


if __name__ == "__main__":
    from sklearn import datasets

    # breast cancer for classification(2 classes)
    x, y = datasets.load_breast_cancer(return_X_y=True)
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    x_train = x[:int(len(x) * 0.8)]
    x_valid = x[int(len(x) * 0.8):]
    y_train = y[:int(len(x) * 0.8)]
    y_valid = y[int(len(x) * 0.8):]
    model = SVM()
    model.fit(x_train, np.expand_dims(y_train, axis=-1), x_valid, np.expand_dims(y_valid, axis=-1))
