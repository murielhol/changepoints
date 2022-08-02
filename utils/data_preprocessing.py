import numpy as np


def get_windows(arr: np.array, window_size: int) -> np.array:
    w_list = list()
    n_records = arr.shape[0]
    num_windows = 1 + int(n_records - window_size)
    for k in range(num_windows):
        w_list.append(arr[k * 1:window_size - 1 + k + 1])
    return np.array(w_list)

