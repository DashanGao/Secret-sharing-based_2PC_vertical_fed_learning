from Client import Client
import numpy as np


class ClientA(Client):
    def __init__(self, name, X, config):
        super().__init__(config, config['ADDR_A'])
        self.name = name
        self.X = X
        self.weights = np.zeros(X.shape[1])  # np.random.rand(X.shape[1]) * 2 - 1
        self.estim_e = None
        self.B_addr = config['ADDR_B']
        self.C_addr = config['ADDR_C']

    def eval(self):
        '''
        A do estimation
        :return: [W_A] * X_A'
        '''
        return self.X.dot(self.weights)

    def predict(self, X):
        return X.dot(self.weights)

    def update_weights(self, estim_error, lr=0.03, lam=0.01):
        self.estim_e = estim_error
        gradient = self.estim_e.dot(self.X) * (2 / self.X.shape[0]) + lam * self.weights
        self.weights -= lr * gradient  # Update encrypted weight using gradient descent

    def set_weights(self, weights):
        self.weights = weights

    def step_1(self):
        try:
            u_a = np.dot(self.weights, self.X.T)
            l_a_1 = np.random.rand(u_a.shape[0])
            u_a_1 = np.random.rand(u_a.shape[0])
            x_a_1 = np.random.rand(self.X.shape[0], self.X.shape[1])
            x_a_2 = self.X - x_a_1
            u_a_2 = u_a - u_a_1
            l_a = u_a ** 2
            l_a_2 = l_a - l_a_1

        except Exception as e:
            print("Wrong 1 in A: %s" % e)

        data_to_C = {"u_a_2": u_a_2, "x_a_2": x_a_2, "l_a_2": l_a_2}
        data_to_B = {"u_a_1": u_a_1, "x_a_1": x_a_1}
        self.data.data.update(data_to_B)
        self.data.data.update(data_to_C)
        self.data.data.update({"u_a": u_a, "l_a": l_a, "l_a_1": l_a_1})

        to_b = [self.B_addr, data_to_B, [self.data.iter_num, 1, str(self.data.iter_num)+"A1"]]
        to_c = [self.C_addr, data_to_C, [self.data.iter_num, 1, str(self.data.iter_num)+"A1"]]

        return to_b, to_c

    def step_2(self):
        try:
            dt = self.data.data
            assert "R" in dt.keys(), "Error: 'R' from C in step 1 not successfully received."
            l_1 = dt["l_a_1"] + 2 * np.dot(dt["u_a_2"], dt["u_b_1"])
        except Exception as e:
            print("A step 2 exception: %s" % e)
            print(dt.keys())
            print(dt['u_a_2'], dt['u_b_1'])
        try:
            u_a_x_a = sum([self.X[i, :] * dt["u_a"][i] for i in range(self.X.shape[0])])
            u_b_1_x_a = sum([self.X[i, :] * dt["u_b_1"][i] for i in range(self.X.shape[0])])
            f_a_1 = 2 * u_a_x_a + 2 * u_b_1_x_a
            u_a_2_x_b_1 = sum([dt["x_b_1"][i, :] * dt["u_a_2"][i] for i in range(self.X.shape[0])])

            f_b_1 = dt["R"] + 2 * u_a_2_x_b_1
        except Exception as e:
            print("A step 2 exception 2 %s " % e)
            print(dt["R"].shape)
            print(dt['u_a_2'].shape, dt['x_b_1'].shape, u_a_2_x_b_1.shape)
            print()
            print(u_a_2_x_b_1)

        self.data.data.update({"f_a_1": f_a_1, "f_b_1": f_b_1})
        return [self.B_addr, {"f_b_1": f_b_1, "l_1": l_1}, [self.data.iter_num, 2, str(self.data.iter_num)+"A2"]]

    def step_3(self):
        dt = self.data.data
        # Compute gradient
        try:
            f_a = (dt["f_a_1"] + dt["f_a_2"] + dt["f_a_3"]) / self.X.shape[0]
            # Update weight
            self.weights = self.weights - self.config["lr"] * f_a - self.config['lambda'] * self.weights
        except Exception as e:
            print("A step 3 exception: %s" % e)
            print(dt.keys())
        print("A weight %d : " % self.data.iter_num, self.weights)
        return
