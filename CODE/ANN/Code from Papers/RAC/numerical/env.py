import numpy as np
from algorithms import vi, shannon, tsallis

class RandomMDP(object):
    def __init__(self, S, A, gamma, lambd=0, algo='shannon'):
        self.S = S
        self.A = A
        self.gamma = gamma
        self.lambd = lambd
        sparse = np.random.binomial(1, 0.05, size=(A*S, S))
        for i in range(A*S):
            if sum(sparse[i, :]) == 0:
                sparse[i, np.random.randint(S)] = 1
        P = np.random.uniform(0, 1, size=(A*S, S))
        self.P = P / np.sum(P, 1)[:, np.newaxis]
        self.R = 0.1 * np.random.uniform(0, 1, size=(A*S, 1))
        self.state = np.random.randint(S)
        self.real_V = self.get_real_V_value(algo)

    def step(self, action):
        p = self.P.reshape(self.S, self.A, self.S)[self.state, action, :]
        reward = self.R.reshape(self.S, self.A)[self.state, action]
        state = self.state = np.random.choice(len(p), p=p)
        return state, reward

    def get_real_V_value(self, algo):
        V0 = np.random.uniform(low=0, high=0.1, size=(self.S, 1))/(1-self.gamma)
        if algo == 'shannon':
            vs = shannon(self.S, self.A, int(1e3), self.P, self.R, V0, self.gamma, self.lambd)
            v = vs[-1]
            #Q = self.R + self.gamma * np.matmul(self.P, v[:, np.newaxis])
            #Q = Q.reshape(self.S, self.A)
        if algo == 'tsallis':
            vs, _ = tsallis(self.S, self.A, int(1e3), self.P, self.R, V0, self.gamma, self.lambd)
            v = vs[-1]
        return v
