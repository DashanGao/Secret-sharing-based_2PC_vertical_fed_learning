from threading import Lock


class DataType:
    def __init__(self):
        self.data = {}
        self.iter_num = 0
        self.step_num = []
        self.received_sender = set()  # "A, B, C"
        self.exit_flag = False
        self.byte_len = 0
        self.lock = Lock()
