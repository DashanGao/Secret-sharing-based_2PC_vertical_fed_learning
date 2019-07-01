import numpy as np
import errno
from threading import Thread, Lock
import time
from contextlib import contextmanager
import os
import json
from utils_ import *
import socket


def server(data_dict, ADDR):
    """
    Listen and connect with clients.
    :param data_dict: store data and exit flag
    :param ADDR: the address (ip, port) of the server
    :return:
    """
    print('Server Run task %s (%s)...' % (ADDR, os.getpid()))
    listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listen_socket.bind(ADDR)
    listen_socket.listen(5)  # REQUEST_QUEUE_SIZE = 3
    # signal.signal(signal.SIGCHLD, grim_reaper)
    while True:
        try:
            # print("Server of %s listening" % name)
            client_connection, client_address = listen_socket.accept()
            t = Thread(target=receive_data, args=(client_connection, data_dict))
            t.start()
        except IOError as e:
            print("Server error !! %s" % e)
            code, msg = e.args
            # restart 'accept' if it was interrupted
            if code == errno.EINTR:
                continue
            else:
                raise


def receive_data(a_socket, data_dict):
    """
    receive data from the connected client, save data to the data_dict.
    :param a_socket: the socket to listen
    :param data_dict: the dict to save raw data reveived and set the exit flag.
    :return:
    """
    buffer = b''
    while True:
        data = a_socket.recv(2048)
        try:
            if not data:
                time.sleep(0.001)
                continue

            while b"$" not in data:
                buffer += data
                try:
                    data = a_socket.recv(2048)
                except Exception as e:
                    print("data receive excetption: %s" % e)

        except Exception as e:
            print("receiver_wrong %s" % e)
            print("data is : %d" % len(buffer))
        try:
            tail = data.split(b'$')
            buffer += tail[0]  # Add tail of a json to the buffer to form a compact json.
            if buffer[-8:] == b'$':
                buffer = buffer[:-8]

            jsons = [buffer]  # List of jsons, each is a compact data package in json format.
            if len(tail) > 2:
                jsons += tail[1:-1]
        except Exception as e:
            print("Excetption:", e)
        # Append broken tail into the data buffer for further data receiving, whether it is empty or not
        if len(tail) >= 2:
            buffer = tail[-1]
        else:
            buffer = b''

        try:
            bytes_len = sum([len(x) for x in jsons])
            jsons = [x.decode("utf-8") for x in jsons]
        except Exception as e:
            print("Data decode error: %s" % e)
            return

        # Iterate each data package (json)
        for data in jsons:
            if data == 'EXIT':  # Judge exit flag here
                data_dict.exit_flag = True
                a_socket.close()
                return
            try:
                received_data = json.loads(data)
            except Exception as e:
                print("received data '%s' can not be loaded as json. \n Error: %s" % (data, e))
                received_data = {}
            try:
                stamp = received_data.pop('stamp')

                # List to numpy array
                for key in received_data.keys():
                    if type(received_data[key]) is list:
                        received_data[key] = np.asarray(received_data[key])

                # Update client data with thread lock.
                if data_dict.lock.acquire():
                    try:
                        data_dict.byte_len += bytes_len
                        data_dict.data.update(received_data)
                        data_dict.received_sender.add(stamp[2])  # mark sender.
                    finally:
                        data_dict.lock.release()
                else:
                    print("Filed to get lock!!!")
                # print(bytes_len)

            except Exception as e:
                print("Exception of stamp %s" % e)

