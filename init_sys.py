import numpy as np

class init_sys:
    def __init__(self, num_bs=10, num_srv=10, max_req=1000, max_srv=2000):
        self.num_bs = num_bs
        self.num_srv = num_srv
        self.max_req = max_req
        self.max_srv = max_srv

    def gen_sys(self):
        srv_rate = np.random.random(self.num_srv) * self.max_srv
        req_rate = np.random.random(self.num_bs) * self.max_req
        return srv_rate, req_rate


    def gen_sys_normal(self):
        #  N(m/2, (m/6)^2)
        srv_rate = np.random.randn(self.num_srv) * self.max_srv / 6 + self.max_srv / 2
        srv_rate[srv_rate < 0] = 0
        req_rate = np.random.randn(self.num_bs) * self.max_req / 6 + self.max_req / 2
        req_rate[req_rate < 0] = 0
        return srv_rate, req_rate
