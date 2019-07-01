from ClientA import ClientA
from ClientB import ClientB
from ClientC import ClientC
from utils_ import *
from time import time, sleep
import os
import json


def process_A(name, data, config):
    """
    The process of client A, where A communicate with client B and C in each iteration to do secure gradient update.
    :param name: the name of client A
    :param data: training dataset of client A
    :param config: the config dict
    :return: True
    """
    print('A Run task %s (%s)...' % (name, os.getpid()))
    client_A = ClientA(name, data, config)
    client_A.set_up()
    client_A.connect([config['ADDR_B'], config['ADDR_C']])

    # print("A flag 0")

    with open("Client_A_log.txt", "a+") as des:
        try:
            des.write(json.dumps(config) + "\n")
        except Exception as e:
            # pass
            print("WRONG ", e)
        # des.write("num_iter\ttime_per_iter\tt_step_1_A\tt_sync_1_A\tt_step_2_A\tt_sync_2_A\tt_step_3_A\tweight_A\n")
        des.write("num_iter\ttime_per_iter\tt_compute_A\tt_sync_A\tweight_A\n")

    start_1 = time()
    compute_t = 0
    sync_t = 0
    byt_len = 0

    while not client_A.data.exit_flag:
        print("A iter num: ", client_A.data.iter_num)
        # Log weight
        if client_A.data.iter_num % 100 == 0:
            log_data("A " + str(client_A.data.iter_num) + " " + str(client_A.weights.tolist()) + "\n", config['A_log_file'])
        t = time()
        # print("A flag 0.5")

        to_b, to_c = client_A.step_1()
        # print("A flag 0.6")

        compute_t += time() - t
        # print("A flag 1")

        t = time()

        client_A.send(to_c[0], to_c[1], to_c[2])
        # print("A flag 2")

        client_A.send(to_b[0], to_b[1], to_b[2])

        # print("A flag 3.")

        client_sync(client_A, ["B", "C"], 1, config)
        sync_t += time() - t
        # print("A flag 3")

        t = time()
        to_b = client_A.step_2()
        compute_t += time() - t
        # print("A flag 4")

        t = time()
        client_A.send(to_b[0], to_b[1], to_b[2])

        client_sync(client_A, ["B", "C"], 2, config)
        sync_t += time() - t
        # print("A flag 5")

        t = time()
        client_A.step_3()
        compute_t += time() - t
        # print("A flag 6")

        byt_len += client_A.data.byte_len
        client_A.data.byte_len = 0

        if client_A.data.iter_num % 10 == 0:
            iter_time = (time() - start_1) / 10
            try:
                with open("Client_A_log_2.txt", "a+") as des:
                    des.write(
                        "%d\t%.5f\t%.5f\t%.5f\t" % (client_A.data.iter_num, iter_time, compute_t / 10, sync_t / 10))
                    des.write(str(client_A.weights.tolist()) + "\n")
            except Exception as e:
                print(e)
            try:
                to_write = str(client_A.data.iter_num) + " " + str(client_A.X.shape) + " " + str(byt_len/10) + "\n"
            except Exception as e:
                print(e)
            with open("Client_A_commu.txt", "a+") as des:
                des.write(to_write)

            compute_t = 0
            sync_t = 0
            start_1 = time()
        # print("A flag 7")

        client_A.data.iter_num += 1
    return True


def process_B(name, X, y, config):
    """
    The process of client A, where A communicate with client B and C in each iteration to do secure gradient update.
    :param name: the name of client B
    :param X: features of the training dataset of client B
    :param y: labels of the training dataset of client B
    :param config: the config dict
    :return: True
    """
    print('B Run task %s (%s)...' % (name, os.getpid()))
    client_B = ClientB(name, X, y, config)
    print(client_B.flag)
    client_B.set_up()
    client_B.connect([config['ADDR_A'], config['ADDR_C']])

    # print("B flag 0")

    with open("Client_B_log.txt", "a+") as des:
        des.write(json.dumps(config) + "\n")
        # des.write("num_iter\ttime_per_iter\tt_step_1_A\tt_sync_1_A\tt_step_2_A\tt_sync_2_A\tt_step_3_A\tweight_A\n")
        des.write("num_iter\ttime_per_iter\tt_compute_A\tt_sync_A\tweight_A\n")

    start_1 = time()
    compute_t = 0
    sync_t = 0

    # print("B flag 1")
    byt_len = 0
    while not client_B.data.exit_flag:
        # Log weight
        if client_B.data.iter_num % 100 == 0:
            log_data("A " + str(client_B.data.iter_num) + " " + str(client_B.weights.tolist()) + "\n", config['B_log_file'])

        t = time()
        # print("B flag 1.5")

        to_a, to_c = client_B.step_1()
        compute_t += time() - t
        # print("B flag 2")
        # print(to_a, "to_C:  ", to_c)

        t = time()
        client_B.send(to_c[0], to_c[1], to_c[2])

        client_B.send(to_a[0], to_a[1], to_a[2])
        # print("B flag 3")

        client_sync(client_B, ["A", "C"], 1, config)
        sync_t += time() - t
        # print("B flag 4")

        t = time()
        to_a = client_B.step_2()
        compute_t += time() - t

        t = time()
        client_B.send(to_a[0], to_a[1], to_a[2])
        client_sync(client_B, ["A", "C"], 2, config)
        sync_t += time() - t
        # print("B flag 5")

        t = time()
        client_B.step_3()
        compute_t += time() - t
        # print("B flag 6")

        byt_len += client_B.data.byte_len
        client_B.data.byte_len = 0

        if client_B.data.iter_num % 10 == 0:
            iter_time = (time() - start_1) / 10
            with open("Client_B_log.txt", "a+") as des:
                des.write("%d\t%.5f\t%.5f\t%.5f\t" % (client_B.data.iter_num, iter_time, compute_t/10, sync_t/10))
                des.write(str(client_B.weights.tolist()) + "\n")
            compute_t = 0
            sync_t = 0
            start_1 = time()
            try:
                to_write = str(client_B.data.iter_num) + " " + str(client_B.X.shape) + " " + str(byt_len/10) + "\n"
            except Exception as e:
                print(e)
            with open("Client_B_commu.txt", "a+") as des:
                des.write(to_write)

        client_B.data.iter_num += 1

    return True


def process_C(name, XA_shape, XB_shape, config):
    """
    The process of client A, where A communicate with client B and C in each iteration to do secure gradient update.
    :param name: The name of the client C
    :param XA_shape: the shape of features of client A
    :param XB_shape: the shape of features of client B
    :param config: the config dict
    :return: True
    """
    print('C Run task %s (%s)...' % ("C", os.getpid()))
    client_C = ClientC(name, XA_shape, XB_shape, config)
    client_C.set_up()
    client_C.connect([config['ADDR_A'], config['ADDR_B']])

    byt_len = 0
    while not client_C.data.exit_flag:

        to_a, to_b = client_C.step_1()
        # print("C flag 1")

        client_C.send(to_a[0], to_a[1], to_a[2])
        client_C.send(to_b[0], to_b[1], to_b[2])
        # print("C flag 1.5")

        client_sync(client_C, ["A", "B"], 1, config)
        # print("C flag 2")

        to_a_, to_b_ = client_C.step_2()
        client_C.send(to_a_[0], to_a_[1], to_a_[2])
        client_C.send(to_b_[0], to_b_[1], to_b_[2])

        byt_len += client_C.data.byte_len
        client_C.data.byte_len = 0

        if client_C.data.iter_num % 10 == 0:
            try:
                to_write = str(client_C.data.iter_num) + " " + str(byt_len/10) + "\n"
                with open("Client_C_commu.txt", "a+") as des:
                    des.write(to_write)
            except Exception as e:
                print(e)
        client_C.data.iter_num += 1
    return True


def client_sync(blocking_client, target_clients, step, config_):
    """
    Block until data received from target partners.
    :param blocking_client: the client who waits for data
    :param target_clients: the clients that should send data to "client"
    :param step: the number of step to wait
    :param config_: the config dict
    :return: True
    """
    itr = str(blocking_client.data.iter_num)
    target_set = set()
    for t in target_clients:
        target_set.add(itr + t + str(step))
    pause_time = config_['pause_time']
    counter = 0
    while not target_set.issubset(blocking_client.data.received_sender):
        sleep(pause_time)
        counter += 1
        if counter == 20:
            print(blocking_client.name, blocking_client.data.received_sender, blocking_client.data.iter_num, step)

    blocking_client.data.received_sender.difference_update(target_set)  # reset list
    return True
