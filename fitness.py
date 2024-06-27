import numpy as np
from math import sqrt


#  providing optimal/minimal delay for one base station without cooperation,
#  requiring srv_rate > 0, req_rate > 0, cloud_delay > 0
def opt_delay(srv_rate, req_rate, cloud_delay=0.1):
    pole = (srv_rate - sqrt(srv_rate / cloud_delay)) / req_rate
    if pole < 0:
        return cloud_delay
    if pole > 1:
        # all requests is processed locally
        return 1 / (srv_rate - req_rate)
    return cloud_delay - (srv_rate * cloud_delay + 1 - sqrt(srv_rate * cloud_delay)) / req_rate


class fitness:
    def __init__(self, srv_rate, req_rate, cloud_delay=0.1):
        self.srv_rate = srv_rate
        self.req_rate = req_rate
        self.num_srv = srv_rate.shape[0]
        self.num_bs = req_rate.shape[0]
        self.cloud_delay = cloud_delay

    # fitness function: avg_delay_greedy
    def fn_old(self, solution):
        delay = np.zeros(self.num_bs, dtype=int) + self.cloud_delay
        srv_bs = np.zeros(self.num_bs, dtype=int)
        for i in range(self.num_srv):
            bs_idx = solution[i]
            srv_bs[bs_idx] = srv_bs[bs_idx] + self.srv_rate[i]
        for i in range(self.num_bs):
            #  1/(mu-lamda) by queue theory
            bs_delay = 1 / (srv_bs[i] - self.req_rate[i])
            if self.cloud_delay > bs_delay > 0:
                delay[i] = bs_delay
        return sum(delay * self.req_rate) / sum(self.req_rate)


    # fitness function: avg_delay_greedy
    def fn_opt(self, solution):
        delay = np.zeros(self.num_bs, dtype=int) + self.cloud_delay
        srv_bs = np.zeros(self.num_bs, dtype=int)
        for i in range(self.num_srv):
            bs_idx = solution[i]
            srv_bs[bs_idx] = srv_bs[bs_idx] + self.srv_rate[i]
        for i in range(self.num_bs):
            delay[i] = opt_delay(srv_bs[i], self.req_rate[i], self.cloud_delay)
        return sum(delay * self.req_rate) / sum(self.req_rate)

    def fn(self, solution):
        return self.fn_opt(solution)
    '''
    from sko.GA import GA
    def ga(self):
        ga_alg = GA(func=self.fitness, n_dim=self.num_site,
                    size_pop=self.num_pop, max_iter=self.max_ite,
                    prob_mut=self.mut_prob,
                    lb=[0] * self.num_site, ub=[1] * self.num_site, precision=[1] * self.num_site)
        best_x, best_f = ga_alg.run()
        return np.array(best_x, dtype=bool), best_f
    '''
