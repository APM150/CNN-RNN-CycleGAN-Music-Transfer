import numpy as np
import datetime


def load_npy_data(npy_data):
    npy_A = np.load(npy_data[0]) * 1.
    npy_B = np.load(npy_data[1]) * 1.
    npy_AB = np.concatenate((npy_A.reshape(npy_A.shape[0], npy_A.shape[1], 1),
                             npy_B.reshape(npy_B.shape[0], npy_B.shape[1], 1)), axis=2)
    return npy_AB


def now() -> str:
    return datetime.datetime.now().strftime(r"%Y_%m_%d_%H_%M_%S")
