import os
import matplotlib.pyplot as plt
import numpy as np
import json
from algorithms import vi, shannon, tsallis, binary_search, cosx, expx, minx, poly, mix
#np.random.seed(seed=3154)
S = 10
A = 5
T0 = 5001
gamma = 0.99

sparse = np.random.binomial(1, 0.05, size=(A*S, S))
for i in range(A*S):
    if sum(sparse[i, :]) == 0:
        sparse[i, np.random.randint(S)] = 1
P = sparse * np.random.uniform(0, 1, size=(A*S, S))
P = P/np.sum(P, 1)[:, np.newaxis]
R = np.random.uniform(low=0, high=1, size=(A*S, 1)) 
V0 = np.random.uniform(low=0, high=0.1, size=(S, 1))/(1-gamma)

print('Now running Value Iteration')
v = vi(S, A, T0, P, R, V0, gamma)
v_true = v[1:]
V = v[-1]
Q = R + gamma * np.matmul(P, V[:, np.newaxis])
Q = Q.reshape(S, A)
count = 0
for i in range(S):
    count += np.sum(Q[i,:]==np.amax(Q[i,:]))
print('Support action space is: ' + str(count/(S*A)))
res = {}
lambds = [0.1]
#lambds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for lambd in lambds:
    print('Now running Shannon Entropy with lambda: ' + str(lambd))
    v, pi, v_real = shannon(S, A, T0, P, R, V0, gamma, lambd, real=True)
    res['logx'] = np.max(np.abs(np.array(v_true[-1]) - np.array(v_real)), axis=1)
    print(v_true[-1])
    print(v_real[-1])
    print('Now running Tsallis with lambda: ' + str(lambd))
    v, pi, v_real = tsallis(S, A, T0, P, R, V0, gamma, lambd, real=True)
    res['tsallis'] = np.max(np.abs(np.array(v_true[-1]) - np.array(v_real)), axis=1)
    print(v_true[-1])
    print(v_real[-1])
    print('Now running cosx with lambda: ' + str(lambd))
    v, pi, v_real = cosx(S, A, T0, P, R, V0, gamma, lambd, real=True)
    res['cosx'] = np.max(np.abs(np.array(v_true[-1]) - np.array(v_real)), axis=1)
    print(v_true[-1])
    print(v_real[-1])
    print('Now running expx with lambda: ' + str(lambd))
    v, pi, v_real = expx(S, A, T0, P, R, V0, gamma, lambd, real=True)
    res['expx'] = np.max(np.abs(np.array(v_true[-1]) - np.array(v_real)), axis=1)
    print(v_true[-1])
    print(v_real[-1])
    print('Now running minx with lambda: ' + str(lambd))
    v, pi, v_real = minx(S, A, T0, P, R, V0, gamma, lambd, real=True)
    res['minx'] = np.max(np.abs(np.array(v_true[-1]) - np.array(v_real)), axis=1)
    print(v_true[-1])
    print(v_real[-1])
    print('Now running poly with lambda: ' + str(lambd))
    v, pi, v_real = poly(S, A, T0, P, R, V0, gamma, lambd, real=True)
    res['poly'] = np.max(np.abs(np.array(v_true[-1]) - np.array(v_real)), axis=1)
    print(v_true[-1])
    print(v_real[-1])
    print('Now running mix with lambda: ' + str(lambd))
    v, pi, v_real = mix(S, A, T0, P, R, V0, gamma, lambd, real=True)
    res['mix'] = np.max(np.abs(np.array(v_true[-1]) - np.array(v_real)), axis=1)
    print(v_true[-1])
    print(v_real[-1])
for key in res.keys():
    plt.plot(np.arange(T0-1), res[key], label=key, linestyle='-')
plt.xlabel('Iteration', fontdict={'size': 20})
plt.ylabel('Error', fontdict={'size': 20})
plt.ylim(0,5)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(loc='lower right', prop={'size':16})
plt.title('Random MDP', fontsize=20)
fig=plt.gcf()
fig.savefig(os.path.join('./', 'error.pdf'), bbox_inches='tight')
from IPython import embed; embed()
