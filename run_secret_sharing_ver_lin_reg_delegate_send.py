from processes_of_clients import *
from utils_ import *
from multiprocessing import Pool
from time import gmtime, strftime
import os


def vertical_secret_sharing_linear_regression(X, y, X_test, y_test, config):
    """
    Start the processes of the three clients: A, B and C.
    :param X: features of the training dataset
    :param y: labels of the training dataset
    :param X_test: features of the test dataset
    :param y_test: labels of the test dataset
    :param config: the config dict
    :return: True
    """
    XA, XB, XA_test, XB_test = vertically_partition_data(X, X_test, config['A_idx'], config['B_idx'])
    print(XA.shape, XB.shape)

    p = Pool(3)  # Init a pool that can run 3 processes concurrently.
    p.apply_async(process_A, args=("A", XA, config))
    p.apply_async(process_B, args=("B", XB, y, config))
    p.apply_async(process_C, args=("C", XA.shape, XB.shape, config))
    print("All process initialized")
    p.close()
    p.join()
    print("All process done.")
    return True


if __name__ == '__main__':
    # Network address config
    HOST = "localhost"
    ADDR_A = (HOST, 4501)
    ADDR_B = (HOST, 4502)
    ADDR_C = (HOST, 4503)
    tm = strftime("_weight_%d_%b_%H_%M_%S.txt", gmtime())
    A_log_file_name = "log_file/A" + tm
    B_log_file_name = "log_file/B" + tm

    if not os.path.exists("log_file"):
        os.makedirs("log_file")

    # Do configuration
    config = {
        'n_clients': 1,
        'key_length': 512,
        'n_iter': 10000,
        'lambda': 0.01,
        'lr': 0.05,
        # 'eta': 0.1,
        'pause_time': 0.001,
        'A_idx': [3, 4, 5, 6, 7, 8],
        'B_idx': [0, 1, 2],
        'A_log_file': A_log_file_name,
        'B_log_file': B_log_file_name,
        'ADDR_A': ADDR_A,
        'ADDR_B': ADDR_B,
        'ADDR_C': ADDR_C,
    }

    # Process data.
    X, y, X_test, y_test = split_train_test()
    print(X.shape, y.shape, X_test.shape, y_test.shape)

    # perform protocol iva secret sharing only.
    vertical_secret_sharing_linear_regression(X, y, X_test, y_test, config)


