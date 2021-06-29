import numpy as np


def to_one_hot(x: np.ndarray):
    if x.ndim != 2 or x.shape[-1] != 1:
        print(f"The shape of input array is illegal, got {x.shape}, expected (samples, 1)")
    tmp_shape = list(x.shape)
    tmp_shape[-1] += x.max(initial=None)
    result = np.zeros(tmp_shape)
    for i in range(x.shape[0]):
        result[i][x[i][0]] = 1
    return result
