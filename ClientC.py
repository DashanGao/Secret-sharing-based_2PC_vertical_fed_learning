from Client import Client
import numpy as np


class ClientC(Client):
    """
    Client C as trusted dealer.
    """
    def __init__(self, name, A_d_shape, B_d_shape, config):
        super().__init__(config, config['ADDR_C'])
        self.name = name
        self.A_data_shape = A_d_shape
        self.B_data_shape = B_d_shape
        self.S = None
        self.R = None
        self.A_addr = config['ADDR_A']
        self.B_addr = config['ADDR_B']

    def step_1(self):
        try:
            S = np.random.rand(self.A_data_shape[1])
            R = np.random.rand(self.B_data_shape[1])
            self.R = R
            self.S = S
        except Exception as e:
            print("C step 1 error 1: %s" % e)

        to_a = [self.A_addr, {"R": R}, [self.data.iter_num, 1, str(self.data.iter_num)+"C1"]]
        to_b = [self.B_addr, {"S": S}, [self.data.iter_num, 1, str(self.data.iter_num) + "C1"]]
        return to_a, to_b

    def step_2(self):
        dt = self.data.data
        l_3 = dt["l_a_2"] + 2 * np.dot(dt["u_a_2"], dt["u_b_2"])
        f_a_3 = 0 - self.S + 2 * sum([dt['x_a_2'][i, :] * dt['u_b_2'][i] for i in range(dt['x_a_2'].shape[0])])
        f_b_3 = 0 - self.R + 2 * sum([dt["x_b_2"][i, :] * dt["u_a_2"][i] for i in range(dt["x_b_2"].shape[0])])
        to_a = [self.A_addr, {"f_a_3": f_a_3}, [self.data.iter_num, 2, str(self.data.iter_num)+"C2"]]
        to_b = [self.B_addr, {"f_b_3": f_b_3, "l_3": l_3}, [self.data.iter_num, 2, str(self.data.iter_num)+"C2"]]
        return to_a, to_b

    def step_3(self):
        pass
