import numpy as np
from sklearn.datasets import load_diabetes
import time
from contextlib import contextmanager
from sklearn.preprocessing import StandardScaler

@contextmanager
def timer():
    """Helper for measuring runtime"""

    time0 = time.perf_counter()
    yield
    print('[elapsed time: %f s]' % (time.perf_counter() - time0))


def split_train_test():
    """
    Import the dataset via sklearn, shuffle and split train/test.
    Return training, target lists for `n_clients` and a holdout test set
    """
    print("Loading data")
    diabetes = load_diabetes()
    y_raw = diabetes.target
    X_raw = diabetes.data
    # print(type(y_raw))
    y_raw = y_raw.reshape(-1, 1)
    # print(y_raw.shape)
    std = StandardScaler()
    std2 = StandardScaler()
    x_scalar = std.fit(X_raw)
    y_scalar = std2.fit(y_raw)
    X = x_scalar.transform(X_raw)
    y = y_scalar.transform(y_raw)
    y = y.reshape(-1)   # important ! shape from (432, 1) to (432, )
    # print(X.shape, y.shape, X.dtype, y.dtype)

    # The features are already preprocessed
    # Shuffle
    perm = np.random.permutation(X.shape[0])
    X, y = X[perm, :], y[perm]

    # Select test at random
    test_size = 50
    test_idx = np.random.choice(X.shape[0], size=test_size, replace=False)
    train_idx = np.ones(X.shape[0], dtype=bool)
    train_idx[test_idx] = False
    X_test, y_test = X[test_idx, :], y[test_idx]
    X_train, y_train = X[train_idx, :], y[train_idx]
    # X_train = np.concatenate((X_train[:300, ], X_train[:300, ]), axis=0)
    # y_train = np.concatenate((y_train[:300], y_train[:300, ]), axis=0)
    return X_train, y_train, X_test, y_test


def vertically_partition_data(X, X_test, A_idx, B_idx):
    """
    Vertically partition feature for party A
    and B
    :param X: train feature
    :param X_test: test feature
    :param A_idx: feature index of party A
    :param B_idx: feature index of party B
    :return: train data for A, B; test data for A, B
    """
    XA = X[:, A_idx]  # Extract A's feature space
    XB = X[:, B_idx]  # Extract B's feature space
    XB = np.c_[XB, np.ones(X.shape[0])]
    XA_test = X_test[:, A_idx]
    XB_test = X_test[:, B_idx]
    XB_test = np.c_[XB_test, np.ones(X_test.shape[0])]
    return XA, XB, XA_test, XB_test


def log_data(data, file_name):
    """
    log data into the given file_name
    :param data: data to be logged
    :param file_name: log file name
    :return: 
    """
    try:
        with open(file_name, "a+") as des:
            des.write(data)
    except Exception as e:
        print(e)
        exit()