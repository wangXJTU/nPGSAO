from fitness import fitness
import numpy as np


class SA(fitness):
    def __init__(self, srv_rate, req_rate, cloud_delay=0.1, dim=20, ite=100, low=0, high=2,
                 T0=100, error=1e-4, k=0.99, step=0.01, L=100):
        super(SA, self).__init__(srv_rate, req_rate, cloud_delay)
        self.dim = dim
        self.low = low
        self.high = high
        self.T0 = T0
        self.error = error
        self.k = k
        self.step = step
        self.L = L
        # 0 ~ high-1
        self.ite = ite
        self.best = None
        self.best_fit = None
        self.history = None

    def sa(self):
        prePoint = np.random.randint(self.low, self.high, size=self.dim)
        curPoint = np.random.randint(self.low, self.high, size=self.dim)
        # fitness, 1 / average delay, greater is better
        preFit = 1 / self.fn(prePoint)
        curFit = 1 / self.fn(curPoint)
        if preFit > curFit:
            self.best = np.copy(prePoint)
            self.best_fit = preFit
            preBestFit = curFit
        else:
            self.best = np.copy(curPoint)
            self.best_fit = curFit
            preBestFit = preFit
        self.history = []
        self.history.append(self.best_fit)
        delta = np.abs(preFit - curFit)
        T = self.T0
        while delta > self.error and T > 0.01:
            T = self.k * T
            for i in range(self.L):
                curPoint = (prePoint + self.step * np.random.randint(self.low, self.high, size=self.dim)).astype(int)
                curPoint = curPoint % (self.high - self.low) + self.low
                curFit = 1 / self.fn(curPoint)
                if curFit > self.best_fit:
                    preBestFit = self.best_fit
                    self.best = np.copy(curPoint)
                    self.best_fit = curFit
                if preFit > curFit:
                    preFit = curFit
                    prePoint = np.copy(curPoint)
                else:
                    prob = np.exp((preFit - curFit) / T)
                    if np.random.rand() < prob:
                        preFit = curFit
            delta = np.abs(self.best_fit - preBestFit)
            #  print(1/self.best_fit, delta, T)
            self.history.append(self.best_fit)
        return 1 / self.best_fit