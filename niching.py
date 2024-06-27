from fitness import fitness
import numpy as np
from sklearn.cluster import KMeans


#  Jaccard distance between two individuals with dim dimensions
def Jaccard(ind1, ind2, dim):
    inter_n = np.sum(ind1 == ind2)
    return inter_n / (dim * 2 - inter_n)


#  calculate similarities between any two individuals of pop
def cal_sim_mat(pop, num_pop, dim, sim_func=Jaccard):
    sim_mat = np.ones(shape=(num_pop, num_pop))
    for i in range(num_pop):
        for j in range(i):
            sim_mat[j][i] = sim_mat[i][j] = sim_func(pop[i], pop[j], dim)
    return sim_mat


def crossing(code1, code2, dim):
    cross_points = np.random.randint(2, size=dim, dtype=bool)
    off1 = code1 * cross_points + code2 * ~cross_points
    off2 = code1 * ~cross_points + code2 * cross_points
    return off1, off2


def mutating(code, dim, low, high):
    points = np.random.randint(2, size=dim, dtype=bool)
    mutated = np.random.randint(low, high, size=dim)
    off = code * points + mutated * ~points
    return off


class nichingswarm(fitness):
    def __init__(self, srv_rate, req_rate, cloud_delay=0.1,
                 num_pop=100, dim=20, w_low=0.4, w_high=1.2, a1=2.0, a2=2.0, ite=100, low=0, high=1,
                 sim_thres=0.5, sim_func=Jaccard):
        super(nichingswarm, self).__init__(srv_rate, req_rate, cloud_delay)
        self.num_pop = num_pop
        self.dim = dim
        # linearly decreasing weight
        self.w_low = w_low
        self.w_high = w_high
        self.a1 = a1
        self.a2 = a2
        self.low = low
        self.high = high - 1
        # 0 ~ high-1
        self.pop = np.random.randint(low, high, size=(num_pop, dim))
        self.vel = np.random.rand(num_pop, dim) * high / 2
        # fitness, 1 / average delay, greater is better
        self.fits = np.zeros(num_pop)
        self.ite = ite
        self.pb = None
        self.pb_fit = None
        #  niche best
        self.nb = None
        self.nb_fit = None
        self.gb = None
        self.gb_fit = None
        self.group = None
        self.num_group = 0
        # matrix
        self.sim_mat = None
        self.sim_thres = sim_thres
        self.sim_func = sim_func
        self.history = None

    def clustering(self):
        # self.num_group = 8
        self.group = np.zeros(shape=(self.num_group, self.num_pop), dtype=bool)
        groups = KMeans(n_clusters=self.num_group, n_init=10).fit(self.pop)
        for i in range(self.num_pop):
            self.group[groups.labels_[i]][i] = True

    '''
    Xinglin Zhang, Jinyi Zhang, Chaoqun Peng, and Xiumin Wang. 2022. 
    Multimodal Optimization of Edge Server Placement Considering System Response Time. 
    ACM Transactions on Sensor Networks, Vol. 19, No. 1, Article 13 (December 2022), 20 pages.
    https://doi.org/10.1145/3534649
    '''

    #  requiring self.sim_mat with similarities between all individuals
    #  providing self.group and self.num_group
    def grouping(self):
        self.group = np.zeros(shape=(self.num_pop, self.num_pop), dtype=bool)
        self.num_group = 0
        for i in range(self.num_pop):
            # largest average similarity of the particle to group, and the group index
            largest_sim = 0
            sim_g = 0
            for g in range(self.num_group):
                #  average similarity of ith particle to gth group
                avg_sim = np.average(self.sim_mat[i][self.group[g]])
                if largest_sim < avg_sim:
                    largest_sim = avg_sim
                    sim_g = g
            #  add ith to sim_gth group
            if largest_sim > self.sim_thres:
                self.group[sim_g][i] = True
            #  add ith to a new group
            else:
                self.group[self.num_group][i] = True
                self.num_group = self.num_group + 1

    def nPSO_org(self, grouping=None):
        if grouping is None:
            grouping = self.grouping
        self.pop = np.random.randint(self.low, self.high, size=(self.num_pop, self.dim))
        self.vel = np.random.rand(self.num_pop, self.dim) * self.high - self.high / 2
        for i in range(self.num_pop):
            self.fits[i] = 1 / self.fn(self.pop[i])
        self.pb = np.copy(self.pop)
        self.pb_fit = np.copy(self.fits)
        best_idx = np.argmax(self.fits)
        self.gb = np.copy(self.pop[best_idx])
        self.gb_fit = self.fits[best_idx]
        self.history = []
        self.history.append(self.gb_fit)

        #
        w_step = (self.w_high - self.w_low) / self.ite
        weight = self.w_high
        for ite in range(self.ite):
            weight = weight - w_step

            r1 = np.random.rand(self.num_pop, self.dim)
            r2 = np.random.rand(self.num_pop, self.dim)
            self.sim_mat = cal_sim_mat(self.pop, self.num_pop, self.dim, self.sim_func)
            grouping()
            self.nb = np.zeros(shape=(self.num_pop, self.dim))
            for g in range(self.num_group):
                # find the best particle in ith group (niche best)
                ng = np.argmax(self.group[g] * self.pb_fit)
                self.nb[self.group[g]] = np.copy(self.pb[ng])

            self.vel = (self.vel * weight + self.a1 * r1 * (self.pb - self.pop)
                        + self.a2 * r2 * (self.nb - self.pop))
            self.vel[self.vel > self.high/2] = self.high/2  # -high/2 ~ high/2
            self.vel[self.vel < -self.high/2] = -self.high/2

            self.pop = (self.pop + self.vel.astype(np.intc)) % (self.high - self.low) + self.low

            for i in range(self.num_pop):
                self.fits[i] = 1 / self.fn(self.pop[i])
                if self.pb_fit[i] < self.fits[i]:
                    self.pb[i] = np.copy(self.pop[i])
                    self.pb_fit[i] = self.fits[i]
            best_idx = np.argmax(self.pb_fit)

            if self.gb_fit < self.pb_fit[best_idx]:
                self.gb = np.copy(self.pb[best_idx])
                self.gb_fit = self.pb_fit[best_idx]
            self.history.append(self.gb_fit)
        return 1 / self.gb_fit

    def nPSO(self):
        return self.nPSO_org()

    def nPSO_cluster(self):
        self.num_group = 5
        return self.nPSO_org(self.clustering)

    # nPGSAO
    def nPGSAO(self, grouping=None):
        if grouping is None:
            grouping = self.grouping
        self.pop = np.random.randint(self.low, self.high, size=(self.num_pop, self.dim))
        for i in range(self.num_pop):
            self.fits[i] = 1 / self.fn(self.pop[i])
        self.pb = np.copy(self.pop)
        self.pb_fit = np.copy(self.fits)
        best_idx = np.argmax(self.fits)
        self.gb = np.copy(self.pop[best_idx])
        self.gb_fit = self.fits[best_idx]
        self.history = []
        self.history.append(self.gb_fit)
        prob_cross = 0.8
        prob_mu = 0.1
        for ite in range(self.ite):
            self.sim_mat = cal_sim_mat(self.pop, self.num_pop, self.dim, self.sim_func)
            grouping()
            self.nb = np.zeros(shape=(self.num_pop, self.dim))
            for g in range(self.num_group):
                # find the best particle in ith group (niche best)
                ng = np.argmax(self.group[g] * self.pb_fit)
                self.nb[self.group[g]] = np.copy(self.pb[ng])

            for i in range(self.num_pop):
                num_offs = 7
                offs = np.zeros(shape=(num_offs, self.dim), dtype=int)
                off_fits = np.zeros(num_offs)
                flag = False
                if np.random.rand() < prob_cross:
                    an_idx = np.random.randint(self.num_pop)
                    offs[0], offs[1] = crossing(self.pop[i], self.pop[an_idx], self.dim)
                    off_fits[0] = 1 / self.fn(offs[0])
                    off_fits[1] = 1 / self.fn(offs[1])
                    flag = True
                if np.random.rand() < prob_cross:
                    offs[2], offs[3] = crossing(self.pop[i], self.pb[i], self.dim)
                    off_fits[2] = 1 / self.fn(offs[2])
                    off_fits[3] = 1 / self.fn(offs[3])
                    flag = True
                if np.random.rand() < prob_cross:
                    offs[4], offs[5] = crossing(self.pop[i], self.nb[i], self.dim)
                    off_fits[4] = 1 / self.fn(offs[4])
                    off_fits[5] = 1 / self.fn(offs[5])
                    flag = True
                if np.random.rand() < prob_mu:
                    offs[6] = mutating(self.pop[i], self.dim, self.low, self.high)
                    off_fits[6] = 1 / self.fn(offs[6])
                    flag = True

                if flag:
                    self.fits[i] = 0
                    self.pop[i] = np.copy(offs[0])
                    for i_off in range(num_offs):
                        if self.fits[i] < off_fits[i_off]:
                            self.pop[i] = np.copy(offs[i_off])
                            self.fits[i] = off_fits[i_off]
            for i in range(self.num_pop):
                self.fits[i] = 1 / self.fn(self.pop[i])
                if self.pb_fit[i] < self.fits[i]:
                    self.pb[i] = np.copy(self.pop[i])
                    self.pb_fit[i] = self.fits[i]
            best_idx = np.argmax(self.pb_fit)
            if self.gb_fit < self.pb_fit[best_idx]:
                self.gb = np.copy(self.pb[best_idx])
                self.gb_fit = self.pb_fit[best_idx]
            self.history.append(self.gb_fit)
        return 1 / self.gb_fit
