from fitness import fitness
import numpy as np
from evolution import GA
import math


class PSO(fitness):
    def __init__(self, srv_rate, req_rate, cloud_delay=0.1,
                 num_pop=100, dim=20, w_low=0.4, w_high=1.2, a1=2.0, a2=2.0, ite=100, prob_mu=0.1, low=0, high=1):
        super(PSO, self).__init__(srv_rate, req_rate, cloud_delay)
        self.num_pop = num_pop
        self.dim = dim
        # linearly decreasing weight
        self.w_low = w_low
        self.w_high = w_high
        self.prob_mu = prob_mu
        self.a1 = a1
        self.a2 = a2
        self.low = low
        self.high = high
        # 0 ~ high-1
        self.pop = None
        self.vel = None  # -high/2 ~ high/2
        # fitness, 1 / average delay, greater is better
        self.fits = np.zeros(num_pop)
        self.ite = ite
        self.pb = None
        self.pb_fit = None
        self.gb = None
        self.gb_fit = None
        self.history = None

    def pso(self):
        self.pop = np.random.randint(self.low, self.high, size=(self.num_pop, self.dim))
        self.vel = np.random.rand(self.num_pop, self.dim) * self.high - self.high / 2
        for i in range(self.num_pop):
            self.fits[i] = 1 / self.fn(self.pop[i])
        self.pb = np.copy(self.pop)
        self.pb_fit = np.copy(self.fits)
        best_idx = np.argmax(self.fits)
        self.gb = self.pop[best_idx]
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
            self.vel = (self.vel * weight + self.a1 * r1 * (self.pb - self.pop)
                        + self.a2 * r2 * (self.gb - self.pop))
            self.vel[self.vel > self.high/2] = self.high/2  # -high/2 ~ high/2
            self.vel[self.vel < -self.high/2] = -self.high/2
            self.pop = (self.pop + self.vel.astype(int)) % (self.high - self.low) + self.low

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

    def pso_sa(self):
        self.pop = np.random.randint(self.low, self.high, size=(self.num_pop, self.dim))
        self.vel = np.random.rand(self.num_pop, self.dim) * self.high - self.high / 2
        for i in range(self.num_pop):
            self.fits[i] = 1 / self.fn(self.pop[i])
        self.pb = np.copy(self.pop)
        self.pb_fit = np.copy(self.fits)
        best_idx = np.argmax(self.fits)
        self.gb = self.pop[best_idx]
        self.gb_fit = self.fits[best_idx]
        self.history = []
        self.history.append(self.gb_fit)

        #
        w_step = (self.w_high - self.w_low) / self.ite
        weight = self.w_high
        T = 100
        T_step = (T - 0.01) / self.ite
        for ite in range(self.ite):
            T = T - T_step
            weight = weight - w_step

            r1 = np.random.rand(self.num_pop, self.dim)
            r2 = np.random.rand(self.num_pop, self.dim)
            self.vel = (self.vel * weight + self.a1 * r1 * (self.pb - self.pop)
                        + self.a1 * r2 * (self.gb - self.pop))
            self.vel[self.vel > self.high/2] = self.high/2  # -high/2 ~ high/2
            self.vel[self.vel < -self.high/2] = -self.high/2

            new_pop = (self.pop + self.vel.astype(int)) % (self.high - self.low) + self.low

            for i in range(self.num_pop):
                new_fit = 1 / self.fn(new_pop[i])
                if new_fit > self.fits[i]:
                    self.pop[i] = np.copy(new_pop[i])
                    self.fits[i] = new_fit
                    if self.pb_fit[i] < self.fits[i]:
                        self.pb[i] = np.copy(self.pop[i])
                        self.pb_fit[i] = self.fits[i]
                else:
                    prob = np.exp(-(new_fit - self.fits[i]) / T)
                    if np.random.rand() < prob:
                        self.pop[i] = np.copy(new_pop[i])
                        self.fits[i] = new_fit

            best_idx = np.argmax(self.pb_fit)
            if self.gb_fit < self.pb_fit[best_idx]:
                self.gb = np.copy(self.pb[best_idx])
                self.gb_fit = self.pb_fit[best_idx]
            self.history.append(self.gb_fit)
        return 1 / self.gb_fit

    def uniform_mutation(self, code):
        points = np.random.randint(2, size=self.dim, dtype=bool)
        mutated = np.random.randint(self.low, self.high, size=self.dim)
        off = code * points + mutated * ~points
        return off

    def psom(self):
        self.pop = np.random.randint(self.low, self.high, size=(self.num_pop, self.dim))
        self.vel = np.random.rand(self.num_pop, self.dim) * self.high - self.high / 2
        for i in range(self.num_pop):
            self.fits[i] = 1 / self.fn(self.pop[i])
        self.pb = np.copy(self.pop)
        self.pb_fit = np.copy(self.fits)
        best_idx = np.argmax(self.fits)
        self.gb = self.pop[best_idx]
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
            self.vel = (self.vel * weight + self.a1 * r1 * (self.pb - self.pop)
                        + self.a2 * r2 * (self.gb - self.pop))
            self.vel[self.vel > self.high/2] = self.high/2  # -high/2 ~ high/2
            self.vel[self.vel < -self.high/2] = -self.high/2
            self.pop = (self.pop + self.vel.astype(int)) % (self.high - self.low) + self.low

            for i in range(self.num_pop):
                if np.random.rand() < self.prob_mu:
                    self.pop[i] = self.uniform_mutation(self.pop[i])
                    self.fits[i] = 1 / self.fn(self.pop[i])
                    if self.pb_fit[i] < self.fits[i]:
                        self.pb[i] = np.copy(self.pop[i])
                        self.pb_fit[i] = self.fits[i]

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


class ABC(fitness):
    def __init__(self, srv_rate, req_rate, cloud_delay=0.1,
                 num_pop=100, dim=20, ite=100, low=0, high=1, limit=5):
        super(ABC, self).__init__(srv_rate, req_rate, cloud_delay)
        self.num_pop = num_pop
        self.dim = dim
        self.low = low
        self.high = high
        self.employed = None  # employed bee, i.e., nectar source
        self.onlooker = None  # onlooker bee
        # fitness, 1 / average delay, greater is better
        self.emp_fits = np.zeros(num_pop)
        self.look_fits = np.zeros(num_pop)
        self.ite = ite
        # recording the time that employed bee has no improvement
        self.unchanged = None
        self.scout_thred = limit
        self.best = None
        self.best_fit = None
        self.history = None

    def abc(self):
        self.employed = np.random.randint(self.low, self.high, size=(self.num_pop, self.dim))
        self.unchanged = np.zeros(self.num_pop, dtype=int)
        self.onlooker = np.zeros(shape=(self.num_pop, self.dim), dtype=int)
        for i in range(self.num_pop):
            self.emp_fits[i] = 1 / self.fn(self.employed[i])
        best_idx = np.argmax(self.emp_fits)
        self.best = np.copy(self.employed[best_idx])
        self.best_fit = self.emp_fits[best_idx]
        self.history = []
        self.history.append(self.best_fit)
        for ite in range(self.ite):
            # Employed Bees Phase
            for i in range(self.num_pop):
                while True:
                    another = np.random.randint(self.num_pop, dtype=int)
                    if another != i:
                        break
                #  [-1, 1)
                rv = np.random.rand() * 2 - 1
                new_employed = self.employed[i] + (rv * (self.employed[i] - self.employed[another])).astype(int)
                new_employed = new_employed % (self.high - self.low) + self.low
                new_fit = 1 / self.fn(new_employed)
                if new_fit > self.emp_fits[i]:
                    self.employed[i] = new_employed
                    self.emp_fits[i] = new_fit
                    self.unchanged[i] = 0
                else:
                    self.unchanged[i] = self.unchanged[i] + 1
            # Onlooker Bees Phase
            acc_fit = np.zeros(self.num_pop)
            acc_fit[0] = self.emp_fits[0]
            for i in range(1, self.num_pop):
                acc_fit[i] = acc_fit[i - 1] + self.emp_fits[i]
            rvf = np.random.rand(self.num_pop) * acc_fit[self.num_pop - 1]
            for i in range(self.num_pop):
                j = 0
                while acc_fit[j] < rvf[i]:
                    j = j + 1

                while True:
                    another = np.random.randint(self.num_pop, dtype=int)
                    if another != j:
                        break
                #  [-1, 1)
                rv = np.random.rand() * 2 - 1
                self.onlooker[i] = self.employed[j] + (rv * (self.employed[j] - self.employed[another])).astype(int)
                self.onlooker[i] = self.onlooker[i] % (self.high - self.low) + self.low
                self.look_fits[i] = 1 / self.fn(self.onlooker[i])
                if self.look_fits[i] > self.emp_fits[j]:
                    self.employed[j] = self.onlooker[i]
                    self.emp_fits[j] = self.look_fits[i]
                    self.unchanged[j] = 0
                else:
                    self.unchanged[j] = self.unchanged[j] + 1
            # Scout Bees Phase
            for i in range(self.num_pop):
                if self.unchanged[i] > self.scout_thred * 2:
                    self.employed[i] = np.random.randint(self.low, self.high, size=self.dim)
            # Memorize the best solution achieved so far
            best_idx = np.argmax(self.emp_fits)

            if self.best_fit < self.emp_fits[best_idx]:
                self.best = np.copy(self.employed[best_idx])
                self.best_fit = self.emp_fits[best_idx]
            self.history.append(self.best_fit)

        return 1 / self.best_fit


class WOA(fitness):
    def __init__(self, srv_rate, req_rate, cloud_delay=0.1,
                 num_pop=100, dim=20, ite=100, low=0, high=1):
        super(WOA, self).__init__(srv_rate, req_rate, cloud_delay)
        self.num_pop = num_pop
        self.dim = dim
        self.low = low
        self.high = high
        self.pop = None
        # fitness, 1 / average delay, greater is better
        self.fits = np.zeros(num_pop)
        self.ite = ite
        self.best = None
        self.best_fit = None
        self.history = None

    # Seyedali Mirjalili and Andrew Lewis,
    # The Whale Optimization Algorithm,
    # Advances in Engineering Software, 2016, 95:51-67
    def woa(self):
        self.pop = np.random.randint(self.low, self.high, size=(self.num_pop, self.dim))
        for i in range(self.num_pop):
            self.fits[i] = 1 / self.fn(self.pop[i])
        best_idx = np.argmax(self.fits)
        self.best = np.copy(self.pop[best_idx])
        self.best_fit = self.fits[best_idx]
        self.history = []
        self.history.append(self.best_fit)

        for ite in range(self.ite):
            #  ICADCML 2023: exponential reduction at the beginning stage and linear reduction at the latter stage
            if ite < self.ite / 2:
                a = np.exp2(1 - ite / self.ite)
            else:
                a = 2 - 2 * ite / self.ite
            # a2 linearly dicreases from -1 to -2 to calculate t in Eq. (3.12)
            a2 = -1 + ite * ((-1) / self.ite)
            for i in range(self.num_pop):
                for j in range(self.dim):
                    if np.random.rand() < 0.5:
                        A = 2 * a * np.random.rand() - a
                        C = 2 * np.random.rand()
                        if A < -1 or A > 1:
                            # 2.2.3. Search for prey (exploration phase)
                            r_idx = np.random.randint(self.num_pop)
                            D = np.abs(C * self.pop[r_idx][j] - self.pop[i][j])
                            self.pop[i][j] = self.pop[r_idx][j] - A * D
                        else:
                            # 2.2.1. Encircling prey
                            D = np.abs(C * self.best[j] - self.pop[i][j])
                            self.pop[i][j] = self.best[j] - A * D
                    else:
                        # 2.2.2. Bubble-net attacking method (exploitation phase)
                        D = np.abs(self.best[j] - self.pop[i][j])
                        l = (a2 - 1) * np.random.rand() + 1
                        self.pop[i][j] = D * np.exp(l) * np.cos(2 * np.pi * l) + self.best[j]

            self.pop[self.pop < self.low] = self.low
            self.pop[self.pop >= self.high] = self.high - 1
            for i in range(self.num_pop):
                self.fits[i] = 1 / self.fn(self.pop[i])
            best_idx = np.argmax(self.fits)
            if self.best_fit < self.fits[best_idx]:
                self.best = np.copy(self.pop[best_idx])
                self.best_fit = self.fits[best_idx]
            self.history.append(self.best_fit)

        return 1 / self.best_fit

    def woa_dim(self):
        self.pop = np.random.randint(self.low, self.high, size=(self.num_pop, self.dim))
        for i in range(self.num_pop):
            self.fits[i] = 1 / self.fn(self.pop[i])
        best_idx = np.argmax(self.fits)
        self.best = np.copy(self.pop[best_idx])
        self.best_fit = self.fits[best_idx]
        self.history = []
        self.history.append(self.best_fit)

        for ite in range(self.ite):
            a = 2 - 2 * ite / self.ite
            # a2 linearly dicreases from -1 to -2 to calculate t in Eq. (3.12)
            a2 = -1 + ite * ((-1) / self.ite)
            for i in range(self.num_pop):
                if np.random.rand() < 0.5:
                    A = 2 * a * np.random.rand() - a
                    C = 2 * np.random.rand()
                    if A < -1 or A > 1:
                        # 2.2.3. Search for prey (exploration phase)
                        r_idx = np.random.randint(self.num_pop)
                        D = np.abs(C * self.pop[r_idx] - self.pop[i])
                        self.pop[i] = self.pop[r_idx] - A * D
                    else:
                        # 2.2.1. Encircling prey
                        D = np.abs(C * self.best - self.pop[i])
                        self.pop[i] = self.best - A * D
                else:
                    # 2.2.2. Bubble-net attacking method (exploitation phase)
                    D = np.abs(self.best - self.pop[i])
                    l = (a2 - 1) * np.random.rand() + 1
                    self.pop[i] = D * np.exp(l) * np.cos(2 * np.pi * l) + self.best

            self.pop[self.pop < self.low] = self.low
            self.pop[self.pop >= self.high] = self.high - 1
            for i in range(self.num_pop):
                self.fits[i] = 1 / self.fn(self.pop[i])
            best_idx = np.argmax(self.fits)
            if self.best_fit < self.fits[best_idx]:
                self.best = np.copy(self.pop[best_idx])
                self.best_fit = self.fits[best_idx]
            self.history.append(self.best_fit)

        return 1 / self.best_fit


class GWO(fitness):
    def __init__(self, srv_rate, req_rate, cloud_delay=0.1,
                 num_pop=100, dim=20, ite=100, low=0, high=1):
        super(GWO, self).__init__(srv_rate, req_rate, cloud_delay)
        self.num_pop = num_pop
        self.dim = dim
        self.low = low
        self.high = high
        self.pop = None
        # fitness, 1 / average delay, greater is better
        self.fits = np.zeros(num_pop)
        self.ite = ite
        # 3 for alpha, beta, and delta wolf
        self.best = None
        self.best_fit = None
        self.history = None

    # Seyedali Mirjalili, Seyed Mohammad Mirjalili and Andrew Lewis,
    # Grey Wolf Optimizer,
    # Advances in Engineering Software, 2014, 69:46-61
    def gwo(self):
        self.pop = np.random.randint(self.low, self.high, size=(self.num_pop, self.dim))
        for i in range(self.num_pop):
            self.fits[i] = 1 / self.fn(self.pop[i])
        # alpha, beta, and delta wolf
        self.best = np.zeros(shape=(3, self.dim), dtype=int)
        self.best_fit = np.zeros(3)
        # alpha
        best_idx = np.argmax(self.fits)
        self.best[0] = np.copy(self.pop[best_idx])
        self.best_fit[0] = self.fits[best_idx]
        self.history = []
        self.history.append(self.best_fit[0])
        # beta
        self.fits[best_idx] = 0
        best_idx = np.argmax(self.fits)
        self.best[1] = np.copy(self.pop[best_idx])
        self.best_fit[1] = self.fits[best_idx]
        # delta
        self.fits[best_idx] = 0
        best_idx = np.argmax(self.fits)
        self.best[2] = np.copy(self.pop[best_idx])
        self.best_fit[2] = self.fits[best_idx]

        for ite in range(self.ite):
            a = 2 - 2 * ite / self.ite
            C = 2 * np.random.rand(self.num_pop)
            for i in range(self.num_pop):
                X = np.zeros(shape=(3, self.dim), dtype=int)
                for k in range(3):
                    A = 2 * a * np.random.rand() - a
                    D = np.abs(C[i] * self.best[k] - self.pop[i])
                    X[k] = self.best[k] - A * D
                self.pop[i] = (X[0] + X[1] + X[2]) / 3

            self.pop[self.pop < self.low] = self.low
            self.pop[self.pop >= self.high] = self.high - 1

            for i in range(self.num_pop):
                self.fits[i] = 1 / self.fn(self.pop[i])
            # alpha, beta, and delta wolf
            # alpha
            best_idx = np.argmax(self.fits)
            self.best[0] = np.copy(self.pop[best_idx])
            self.best_fit[0] = self.fits[best_idx]
            if self.history[-1] < self.best_fit[0]:
                self.history.append(self.best_fit[0])
            else:
                self.history.append(self.history[-1])
            # beta
            self.fits[best_idx] = 0
            best_idx = np.argmax(self.fits)
            self.best[1] = np.copy(self.pop[best_idx])
            self.best_fit[1] = self.fits[best_idx]
            # delta
            self.fits[best_idx] = 0
            best_idx = np.argmax(self.fits)
            self.best[2] = np.copy(self.pop[best_idx])
            self.best_fit[2] = self.fits[best_idx]

        return 1 / self.history[-1]


# HEIDARI A A, MIRJALILI S, FARIS H, et al.
# Harris hawks optimization: algorithm and applications[J].
# Future Generation Computer Systems, 2019, 97: 849-872.
class HHO(fitness):
    def __init__(self, srv_rate, req_rate, cloud_delay=0.1,
                 num_pop=100, dim=20, ite=100, low=0, high=1):
        super(HHO, self).__init__(srv_rate, req_rate, cloud_delay)
        self.num_pop = num_pop
        self.dim = dim
        self.low = low
        self.high = high
        # fitness, 1 / average delay, greater is better
        self.fits = np.zeros(num_pop)
        self.ite = ite
        # rabbit
        self.best = None
        self.best_fit = None
        self.history = None

    def hho(self):
        self.pop = np.random.randint(self.low, self.high, size=(self.num_pop, self.dim))
        for i in range(self.num_pop):
            self.fits[i] = 1 / self.fn(self.pop[i])
        best_idx = np.argmax(self.fits)
        self.best = np.copy(self.pop[best_idx])
        self.best_fit = self.fits[best_idx]
        self.history = []
        self.history.append(self.best_fit)

        #  for the Lévy flight function
        beta = 1.5
        sigma = math.pow((math.gamma(1 + beta) * math.sin(beta * math.pi / 2))
                         / (math.gamma((1 + beta) / 2) * beta * math.pow(2, (beta - 1) / 2)),
                         1 / beta)

        for ite in range(self.ite):
            E = 2 * (2 * np.random.rand(self.num_pop) - 1) * (1 - ite / self.ite)
            J = 2 * (1 - np.random.rand(self.num_pop))
            Xmean = np.mean(self.pop, axis=0)
            for i in range(self.num_pop):
                Eabs = math.fabs(E[i])
                if Eabs >= 1:
                    if np.random.rand() >= 0.5:
                        while True:
                            r_idx = np.random.randint(self.num_pop)
                            if r_idx != i:
                                break
                        self.pop[i] = (self.pop[r_idx] - np.random.rand() * np.abs(
                            self.pop[r_idx] - 2 * np.random.rand() * self.pop[i])).astype(int)
                    else:
                        self.pop[i] = (self.best - Xmean - np.random.rand() * (
                                self.low + np.random.rand() * (self.high - self.low))).astype(int)
                else:
                    r = np.random.rand()
                    if Eabs >= 0.5 and r >= 0.5:
                        self.pop[i] = (self.best - self.pop[i] - E[i] * np.abs(J[i] * self.best
                                                                               - self.pop[i])).astype(int)
                    elif Eabs < 0.5 <= r:
                        self.pop[i] = (self.best - E[i] * np.abs(self.best - self.pop[i])).astype(int)
                    else:
                        if Eabs >= 0.5:
                            Y = (self.best - E[i] * np.abs(J[i] * self.best - self.pop[i])).astype(int)
                        else:
                            Y = (self.best - E[i] * np.abs(J[i] * self.best - Xmean)).astype(int)
                        Y = Y % (self.high - self.low) + self.low
                        fit = 1 / self.fn(Y)
                        if fit > self.fits[i]:
                            self.pop[i] = np.copy(Y)
                            self.fits[i] = fit
                        else:
                            Z = (Y + np.random.rand(self.dim) * 0.01 * np.random.rand(self.dim) * sigma
                                 / np.power(np.random.rand(self.dim), 1 / beta)).astype(int)
                            Z = Z % (self.high - self.low) + self.low
                            fit = 1 / self.fn(Z)
                            if fit > self.fits[i]:
                                self.pop[i] = np.copy(Z)
                                self.fits[i] = fit
            self.pop = self.pop % (self.high - self.low) + self.low
            for i in range(self.num_pop):
                self.fits[i] = 1 / self.fn(self.pop[i])
            best_idx = np.argmax(self.fits)
            if self.best_fit < self.fits[best_idx]:
                self.best = np.copy(self.pop[best_idx])
                self.best_fit = self.fits[best_idx]
            self.history.append(self.best_fit)
        return 1 / self.best_fit
