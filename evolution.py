from fitness import fitness
import numpy as np


class GA(fitness):
    def __init__(self, srv_rate, req_rate, cloud_delay=0.1,
                 num_pop=100, dim=20, prob_cross=0.8, prob_mu=0.1, ite=100, low=0, high=2):
        super(GA, self).__init__(srv_rate, req_rate, cloud_delay)
        self.num_pop = num_pop
        self.dim = dim
        self.prob_cross = prob_cross
        self.prob_mu = prob_mu
        self.low = low
        self.high = high
        # 0 ~ high-1
        self.pop = np.random.randint(low, high, size=(num_pop, dim))
        # fitness, 1 / average delay, greater is better
        self.fits = np.zeros(num_pop)
        self.ite = ite
        self.best = None
        self.best_fit = None
        self.history = None

    def single_cross(self, code1, code2):
        cross_point = np.random.randint(self.dim)
        off1 = np.append(code1[:cross_point], code2[cross_point:])
        off2 = np.append(code2[:cross_point], code1[cross_point:])
        return off1, off2

    def double_cross(self, code1, code2):
        cross_points = np.random.randint(self.dim, size=2)
        if cross_points[0] > cross_points[1]:
            p1 = cross_points[1]
            p2 = cross_points[0]
        else:
            p1 = cross_points[0]
            p2 = cross_points[1]

        off1 = np.concatenate((code1[:p1], code2[p1:p2], code1[p2:]))
        off2 = np.concatenate((code2[:p1], code1[p1:p2], code2[p2:]))
        return off1, off2

    def uniform_cross(self, code1, code2):
        cross_points = np.random.randint(2, size=self.dim, dtype=bool)
        off1 = code1 * cross_points + code2 * ~cross_points
        off2 = code1 * ~cross_points + code2 * cross_points
        return off1, off2

    def uniform_mutation(self, code):
        points = np.random.randint(2, size=self.dim, dtype=bool)
        mutated = np.random.randint(self.low, self.high, size=self.dim)
        off = code * points + mutated * ~points
        return off

    #  roulette wheel selection
    def roulette(self, fn):
        sum_fn = np.sum(fn)
        rv = np.random.uniform() * sum_fn
        cal_fn = fn[0]
        i = 0
        while rv > cal_fn:
            i = i + 1
            cal_fn = cal_fn + fn[i]
        return i-1

    #  tournament selection
    def tournament(self, fn, sample=4):
        num = fn.shape[0]
        inds_idx = np.random.randint(num, size=sample)
        inds = fn[inds_idx]
        max_idx = np.argmax(inds)
        return inds_idx[max_idx]

    def ga(self, crossing=None, mutating=None, selecting=None):
        if crossing is None:
            crossing = self.uniform_cross
        if mutating is None:
            mutating = self.uniform_mutation
        if selecting is None:
            selecting = self.roulette

        self.pop = np.random.randint(self.low, self.high, size=(self.num_pop, self.dim))
        for i in range(self.num_pop):
            self.fits[i] = 1 / self.fn(self.pop[i])
        best_idx = np.argmax(self.fits)
        self.best = self.pop[best_idx]
        self.best_fit = self.fits[best_idx]
        self.history = []
        self.history.append(self.best_fit)

        for ite in range(self.ite):
            # print("%dth iteration" % ite)
            #  offsprings
            offs = None
            '''crossover'''
            for i in range(self.num_pop):
                if np.random.rand() < self.prob_cross:
                    an_idx = np.random.randint(self.num_pop)
                    off1, off2 = crossing(self.pop[i], self.pop[an_idx])
                    if offs is None:
                        offs = np.vstack((off1, off2))
                    else:
                        offs = np.concatenate((offs, np.vstack((off1, off2))))

            '''mutation'''
            for i in range(self.num_pop):
                if np.random.rand() < self.prob_mu:
                    off1 = mutating(self.pop[i])
                    if offs is None:
                        offs = np.array([off1])
                    else:
                        offs = np.concatenate((offs, np.array([off1])))

            '''select'''
            pop_idx = int(self.num_pop * 0.1)
            sort_idx = np.argsort(-self.fits)
            #  reserve top 10% of individuals
            self.pop[:pop_idx] = self.pop[sort_idx[:pop_idx]]
            self.fits[:pop_idx] = self.fits[sort_idx[:pop_idx]]
            #  evaluating fitness of offsprings
            num_off = offs.shape[0]
            fits_off = np.zeros(num_off)
            # print(offs)
            for i in range(num_off):
                fits_off[i] = 1 / self.fn(offs[i])
            #  update the best individual
            best_idx = np.argmax(fits_off)
            if fits_off[best_idx] > self.best_fit:
                self.best = np.copy(offs[best_idx])
                self.best_fit = fits_off[best_idx]
            #  select offsprings to form the next generation with num_pop individuals
            while pop_idx < self.num_pop:
                new_idx = selecting(fits_off)
                self.pop[pop_idx] = offs[new_idx]
                pop_idx = pop_idx + 1

            self.history.append(self.best_fit)

        return 1 / self.best_fit


    #  pso ga
    def pgsao(self, crossing=None, mutating=None):
        if crossing is None:
            crossing = self.uniform_cross
        if mutating is None:
            mutating = self.uniform_mutation

        self.pop = np.random.randint(self.low, self.high, size=(self.num_pop, self.dim))
        for i in range(self.num_pop):
            self.fits[i] = 1 / self.fn(self.pop[i])
        pb = np.copy(self.pop)
        pb_fit = np.copy(self.fits)
        best_idx = np.argmax(self.fits)
        self.best = self.pop[best_idx]
        self.best_fit = self.fits[best_idx]
        self.history = []
        self.history.append(self.best_fit)

        for ite in range(self.ite):
            for i in range(self.num_pop):
                num_offs = 7
                offs = np.zeros(shape=(num_offs, self.dim), dtype=int)
                off_fits = np.zeros(num_offs)
                flag = False
                if np.random.rand() < self.prob_cross:
                    an_idx = np.random.randint(self.num_pop)
                    offs[0], offs[1] = crossing(self.pop[i], self.pop[an_idx])
                    off_fits[0] = 1 / self.fn(offs[0])
                    off_fits[1] = 1 / self.fn(offs[1])
                    flag = True
                if np.random.rand() < self.prob_cross:
                    offs[2], offs[3] = crossing(self.pop[i], pb[i])
                    off_fits[2] = 1 / self.fn(offs[2])
                    off_fits[3] = 1 / self.fn(offs[3])
                    flag = True
                if np.random.rand() < self.prob_cross:
                    offs[4], offs[5] = crossing(self.pop[i], self.best)
                    off_fits[4] = 1 / self.fn(offs[4])
                    off_fits[5] = 1 / self.fn(offs[5])
                    flag = True
                if np.random.rand() < self.prob_mu:
                    offs[6] = mutating(self.pop[i])
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
                if pb_fit[i] < self.fits[i]:
                    pb[i] = np.copy(self.pop[i])
                    pb_fit[i] = self.fits[i]
            best_idx = np.argmax(pb_fit)
            if self.best_fit < pb_fit[best_idx]:
                self.best = np.copy(pb[best_idx])
                self.best_fit = pb_fit[best_idx]
            self.history.append(self.best_fit)

        return 1 / self.best_fit
