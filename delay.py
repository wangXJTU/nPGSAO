import numpy as np


def avg_delay_greedy(solution, srv_rate, req_rate, cloud_delay):
    num_bs = req_rate.shape[0]
    num_srv = srv_rate.shape[0]
    delay = np.zeros(num_bs, dtype=int) + cloud_delay
    for i in range(num_srv):
        bs_idx = solution[i]
        if srv_rate[i] > req_rate[bs_idx]:
            #  1/(mu-lamda) by queue theory
            bs_delay = 1 / (srv_rate[i] - req_rate[bs_idx])
            if cloud_delay > bs_delay:
                delay[bs_idx] = bs_delay
    return sum(delay * req_rate) / sum(req_rate)


def avg_delay_best(solution, srv_rate, req_rate, cloud_delay):
    num_bs = req_rate.shape[0]
    num_srv = srv_rate.shape[0]
    delay = np.zeros(num_bs, dtype=int) + cloud_delay
    for i in range(num_srv):
        bs_idx = solution[i]
        if srv_rate[i] > req_rate[bs_idx]:
            #  1/(mu-lamda) by queue theory
            bs_delay = 1 / (srv_rate[i] - req_rate[bs_idx])
            if cloud_delay < bs_delay:
                delay[bs_idx] = bs_delay
    return sum(delay * req_rate) / sum(req_rate)
