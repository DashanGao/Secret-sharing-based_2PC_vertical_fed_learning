import numpy as np
import errno
from threading import Thread, Lock
import time
from contextlib import contextmanager
import json
from utils_ import *
import socket
import DataType
import Server


class Client:
    def __init__(self, config, addre):
        self.data = DataType.DataType()
        self.flag = True
        self.config = config
        self.ADDR = addre
        self.name = ""
        self.serve = None
        self.connections = {}

    def connect(self, addrs_to_connect):
        for addr in addrs_to_connect:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect(addr)
            self.connections[addr] = s
            # print("%s connected to %s" % (self.name, str(addr)))

    def set_up(self):
        self.serve = Thread(target=Server.server, args=(self.data, self.ADDR))
        self.serve.start()
        print("%s setting up listening server successfully!" % self.name)

    def get_data(self):
        return self.data.data

    def set_data(self, data_):
        self.data.data = data_

    def send(self, receiver, data_, stamp):
        # Receiver is the address of another party
        assert type(data_) is dict
        # Serialize numpy.ndarray
        for key in data_.keys():
            if type(data_[key]) is np.ndarray:
                data_[key] = data_[key].tolist()

        data_['stamp'] = stamp  # stamp: [#iter, #step, sender]
        data = json.dumps(data_)  # Serialize data

        # send data to the target party
        try:
            self.connections[receiver].send(data.encode("utf-8"))
        except Exception as e:
            print("Sending error: %s" % e)

        # Terminate data transmission by sending '$'
        try:
            self.connections[receiver].send(b'$')
        except Exception as e:
            print("Ending flag error: %s" % e)

