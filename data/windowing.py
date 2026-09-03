# The windowing logic is by sliding a window of length 3 (condition days input for forecasting future)
# It also contains normalisation helper functions.

import numpy as np
from config import COND_DAYS, GEN_DAYS, TRAIN_SPLIT


def build_windows(data, window_cond=COND_DAYS, window_gen=GEN_DAYS):
    total_window = window_cond + window_gen
    X_cond, Y_gen = [], []
    for i in range(len(data) - total_window + 1):
        X_cond.append(data[i: i + window_cond])
        Y_gen.append(data[i + window_cond: i + total_window])
    return np.array(X_cond), np.array(Y_gen)


def train_test_split(X_cond, Y_gen, train_split=TRAIN_SPLIT):
    n = len(X_cond)
    train_end = int(n * train_split)
    X_train, Y_train = X_cond[:train_end], Y_gen[:train_end]
    X_test, Y_test = X_cond[train_end:], Y_gen[train_end:]
    return X_train, Y_train, X_test, Y_test


def compute_normalization_stats(X_train):
    return X_train.mean(), X_train.std()


def normalize(arr, mean, std):
    if std == 0:
        raise ValueError("normalization std is zero — cannot normalize constant data")
    return (arr - mean) / std


def denormalize(arr, mean, std):
    return arr * std + mean