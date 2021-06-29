import numpy as np


class NaiveBayes:
    def __init__(self):
        # 存储样本数量
        self.__m = None
        # 存储属性竖向
        self.__n = None
        # This will be a dictionary, whose keys refer to label and values refer to attribute dicts
        # label: any -> attributes: dict
        self.__label_to_attributes: dict = dict()
        # label: any -> num: int:
        self.__label_to_num: dict = dict()

    def __init_database(self, attributes: np.ndarray, labels: np.ndarray):
        self.__label_to_attributes.clear()
        self.__label_to_num.clear()

        for i in range(self.__m):
            # get tmp label
            tmp_label = labels[i][0]
            if tmp_label not in self.__label_to_attributes.keys():
                self.__label_to_attributes[tmp_label] = dict()
                self.__label_to_num[tmp_label] = 1
            else:
                self.__label_to_num[tmp_label] += 1
                for j in range(self.__n):
                    # get tmp attribute
                    tmp_attribute = attributes[i][j]
                    if tmp_attribute not in self.__label_to_attributes[tmp_label].keys():
                        self.__label_to_attributes[tmp_label][tmp_attribute] = 1
                    else:
                        self.__label_to_attributes[tmp_label][tmp_attribute] += 1

    def predict(self, x: np.ndarray, more_details: bool = False):
        if x.ndim != 2 or x.shape[-1] != self.__n:
            raise ValueError(
                f"The array needed to be predicted is illegal, got shape {x.shape}, expected (samples, {self.__n})")
        predictions: list = list()
        for i in range(x.shape[0]):
            predicted_label = None
            max_value: float = -np.inf
            for tmp_label in self.__label_to_attributes.keys():
                tmp_label_num = self.__label_to_num[tmp_label]
                # get tmp label number
                # calculate p(c)
                result = tmp_label_num / self.__m
                print(f"\tp({tmp_label}) = {result}")
                # calculate p(a1|c)p(a2|c)...p(an|c)
                for j in range(self.__n):
                    p_attribute = self.__label_to_attributes[tmp_label][x[i][j]] / tmp_label_num
                    result *= p_attribute
                    print(f"\tp({x[i][j]}|{tmp_label}) = {p_attribute}")
                # argmax
                if result >= max_value:
                    max_value, predicted_label = result, tmp_label
                if more_details:
                    print(f"For label {tmp_label}, the c({tmp_label}) = {result}\n")
            if more_details:
                print(f"Choose label {predicted_label}, which got max value {max_value}")
            predictions.append(predicted_label)
        return np.expand_dims(np.array(predictions), axis=-1)

    def fit(self, x: np.ndarray,
            y: np.ndarray,
            verbose: int = 1):
        if x.ndim != 2:
            raise ValueError(
                f"The shape of the training set is illegal, got {x.shape}, expected [samples, features]")
        if y.ndim != 2 or y.shape[-1] != 1:
            raise ValueError(
                f"The shape of the training label is illegal, got {y.shape}, expected [samples, 1]")
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"The samples of x and y must be equal, got x {x.shape}, y {y.shape}")
        self.__m, self.__n = x.shape[0], x.shape[1]

        self.__init_database(x, y)
        if verbose > 0:
            print("The statistics are shown by follow:")
            for key, value in self.__label_to_num.items():
                print(f"Class label: {key}, label number: {value}")
            print(f"Used samples {self.__m}, found {self.__n} attributes\n")
        return

    def score(self, x: np.ndarray, y: np.ndarray):
        predictions = self.predict(x)
        acc = np.count_nonzero(predictions == y) / x.shape[0]
        print(f"Final score, accuracy {acc}")

    def run_sample(self, name: str = 'watermelon') -> None:
        """
        :param name: 'play_tennis' or 'watermelon' are both accepted
        :return: None
        """
        from sklearn import model_selection
        if name == 'play_tennis':
            x_data: np.ndarray = np.array([
                ["Sunny", "Hot", "High", "Weak"],
                ["Sunny", "Hot", "High", "Strong"],
                ["Overcast", "Hot", "High", "Weak"],
                ["Rain", "Mild", "High", "Weak"],
                ["Rain", "Cool", "Normal", "Weak"],
                ["Rain", "Cool", "Normal", "Strong"],
                ["Overcast", "Cool", "Normal", "Strong"],
                ["Sunny", "Mild", "High", "Weak"],
                ["Sunny", "Cool", "Normal", "Weak"],
                ["Rain", "Mild", "Normal", "Weak"],
                ["Sunny", "Mild", "Normal", "Strong"],
                ["Overcast", "Mild", "High", "Strong"],
                ["Overcast", "Hot", "Normal", "Weak"],
                ["Rain", "Mild", "High", "Strong"]
            ])
            y_data: np.ndarray = np.expand_dims(np.array(
                ["No", "No", "Yes", "Yes", "Yes", "No", "Yes", "No", "Yes", "Yes", "Yes", "Yes", "Yes", "No"]
            ), axis=-1)
            self.fit(x_data, y_data)
            print(self.predict(np.array([["Sunny", "Cool", "High", "Strong"]]), more_details=True))
        elif name == 'watermelon':
            x_data: np.ndarray = np.array([
                ["青绿", "蜷缩", "浊响", "清晰", "凹陷", "硬滑"],
                ["乌黑", "蜷缩", "沉闷", "清晰", "凹陷", "硬滑"],
                ["乌黑", "蜷缩", "浊响", "清晰", "凹陷", "硬滑"],
                ["青绿", "蜷缩", "沉闷", "清晰", "凹陷", "硬滑"],
                ["浅白", "蜷缩", "浊响", "清晰", "凹陷", "硬滑"],
                ["青绿", "稍蜷", "浊响", "清晰", "稍凹", "软粘"],
                ["乌黑", "稍蜷", "浊响", "稍糊", "稍凹", "软粘"],
                ["乌黑", "稍蜷", "浊响", "清晰", "稍凹", "硬滑"],
                ["乌黑", "稍蜷", "沉闷", "稍糊", "稍凹", "硬滑"],
                ["青绿", "硬挺", "清脆", "清晰", "平坦", "软粘"],
                ["浅白", "硬挺", "清脆", "模糊", "平坦", "硬滑"],
                ["浅白", "蜷缩", "浊响", "模糊", "平坦", "软粘"],
                ["青绿", "稍蜷", "浊响", "稍糊", "凹陷", "硬滑"],
                ["浅白", "稍蜷", "沉闷", "稍糊", "凹陷", "硬滑"],
                ["乌黑", "稍蜷", "浊响", "清晰", "稍凹", "软粘"],
                ["浅白", "蜷缩", "浊响", "模糊", "平坦", "硬滑"],
                ["青绿", "蜷缩", "沉闷", "稍糊", "稍凹", "硬滑"],
            ])
            y_data: np.ndarray = np.expand_dims(np.array(
                ["是", "是", "是", "是", "是", "是", "是", "是", "否", "否", "否", "否", "否", "否", "否", "否", "否"]
            ), axis=-1)
            self.fit(x_data, y_data)
            print(self.predict(np.array([["浅白", "稍蜷", "浊响", "清晰", "稍凹", "硬滑"]]), more_details=True))
        else:
            return


"""Run Sample"""
if __name__ == "__main__":
    model = NaiveBayes()
    model.run_sample('play_tennis')
    model.run_sample('watermelon')
