from Client import Client
import numpy as np


class ClientB(Client):
    def __init__(self, name, X, y, config):
        super().__init__(config, config['ADDR_B'])
        self.name = name
        self.X = X
        self.y = y
        self.weights = np.zeros(X.shape[1])  # np.random.rand(X.shape[1]) * 2 - 1
        self.sum_X = np.sum(X, 0)
        self.estim_e = None
        self.loss = []
        self.A_addr = config["ADDR_A"]
        self.C_addr = config['ADDR_C']

    def estim_error(self, a_eval):
        '''
        B compute the estimation error (h(X) - y) for further gradient computation.
        :return: [W_B, b] * [X_B, 1]' + [W_A] * X_A' - y
        '''
        b_eval = self.eval()
        try:
            self.estim_e = b_eval + a_eval - self.y
        except Exception as e:
            print(e)
            self.estim_e = np.zeros(self.y.shape)
        return self.estim_e

    def eval(self):
        '''
        B do estimation
        :return: [W_B] * X_B'
        '''
        return self.X.dot(self.weights)

    def predict(self, X):
        return X.dot(self.weights)

    def update_weights(self, lr=0.03, lam=0.01):
        if self.estim_e is None:
            return
        gradient = self.estim_e.dot(self.X) * (2 / self.X.shape[0]) + lam * self.weights
        self.weights -= lr * gradient  # Update encrypted weight using gradient descent

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights

    def step_1(self):
        x_b_1 = np.random.rand(self.X.shape[0], self.X.shape[1])
        u_b = np.dot(self.weights, self.X.T) - self.y
        u_b_1 = np.random.rand(u_b.shape[0])
        u_b_2 = u_b - u_b_1

        x_b_2 = self.X - x_b_1

        to_a = [self.A_addr, {"x_b_1": x_b_1, "u_b_1": u_b_1}, [self.data.iter_num, 1, str(self.data.iter_num)+"B1"]]
        to_c = [self.C_addr, {"x_b_2": x_b_2, "u_b_2": u_b_2}, [self.data.iter_num, 1, str(self.data.iter_num)+"B1"]]
        self.data.data.update({"x_b_2": x_b_2, "u_b_2": u_b_2, "x_b_1": x_b_1, "u_b_1": u_b_1, "u_b": u_b})
        return to_a, to_c

    def step_2(self):
        dt = self.data.data
        l_2 = dt["u_b"] ** 2 + 2 * np.dot(dt["u_b"], dt["u_a_1"])
        f_b_2 = 2 * sum(
            [self.X[i, :] * dt["u_b"][i] for i in range(self.X.shape[0])]) + 2 * \
                sum([self.X[i, :] * dt["u_a_1"][i] for i in range(self.X.shape[0])])
        try:
            f_a_2 = dt["S"] + 2 * sum([dt["x_a_1"][i, :] * dt["u_b_2"][i] for i in range(self.X.shape[0])])
        except Exception as e:
            print("B step 2 exception: %s" % e)
        self.data.data.update({"l_2": l_2, "f_b_2": f_b_2, "f_a_2": f_a_2})
        return [self.A_addr, {"f_a_2": f_a_2}, [self.data.iter_num, 2, str(self.data.iter_num)+"B2"]]

    def step_3(self):
        dt = self.data.data
        f_b = (dt["f_b_1"] + dt["f_b_2"] + dt["f_b_3"]) / self.X.shape[0]
        l = (dt["l_1"] + dt['l_2'] + dt['l_3']) / self.X.shape[0]
        self.weights = self.weights - self.config["lr"] * f_b - self.config["lambda"] * self.weights
        print("B weight %d: " % self.data.iter_num, self.weights)
        self.loss.append(l)
        return
