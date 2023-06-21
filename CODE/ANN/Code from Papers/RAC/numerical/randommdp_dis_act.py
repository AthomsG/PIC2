import os
import matplotlib.pyplot as plt
import numpy as np
import json
from algorithms import vi, shannon, tsallis, binary_search, cosx, expx, minx, poly, mix
np.random.seed(seed=3154)
S = 50
A = 10
T0 = 500
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
V = v[-1]
Q = R + gamma * np.matmul(P, V[:, np.newaxis])
Q = Q.reshape(S, A)
count = 0
for i in range(S):
    count += np.sum(Q[i,:]==np.amax(Q[i,:]))
print('Support action space is: ' + str(count/(S*A)))
lambds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
res = {'tsallis': {'lambdas': lambds}, 'cosx': {'lambdas': lambds}, 'expx': {'lambdas': lambds}, 
        'logx': {'lambdas': lambds}, 'min': {'lambdas': lambds}, 'poly': {'lambdas': lambds},
        'mix': {'lambdas': lambds}}
for lambd in lambds:
    print('Now running Logx with lambda: ' + str(lambd))
    v, Pi = shannon(S, A, T0, P, R, V0, gamma, lambd)
    count = 0
    for j in range(A):
        if lambd == lambds[0]:
            res['logx'][j] = [Pi[0, j]]
        else:
            res['logx'][j].append(Pi[0, j])
    for i in range(S):
        count += np.sum(Pi[i, :]>0)

    if lambd == lambds[0]:
        res['logx']['act_ratio'] = [count/(S*A)]
    else:
        res['logx']['act_ratio'].append(count/(S*A))
    print('Support action space is: ' + str(count/(S*A)))

    print('Now running Cosx with lambda: ' + str(lambd))
    v, Pi = cosx(S, A, T0, P, R, V0, gamma, lambd)
    for i in range(S):
        p = Pi[i, :]
        p[p<=1e-4/A] = 0
        Pi[i, :] = p
    count = 0
    for j in range(A):
        if lambd == lambds[0]:
            res['cosx'][j] = [Pi[0, j]]
        else:
            res['cosx'][j].append(Pi[0, j])
    for i in range(S):
        count += np.sum(Pi[i, :]>0)

    if lambd == lambds[0]:
        res['cosx']['act_ratio'] = [count/(S*A)]
    else:
        res['cosx']['act_ratio'].append(count/(S*A))
    print('Support action space is: ' + str(count/(S*A)))
    
    print('Now running Tsallis with lambda: ' + str(lambd))
    v, Pi = tsallis(S, A, T0, P, R, V0, gamma, lambd)
    for i in range(S):
        p = Pi[i, :]
        p[p<=1e-4/A] = 0
        Pi[i, :] = p
    count = 0
    for j in range(A):
        if lambd == lambds[0]:
            res['tsallis'][j] = [Pi[0, j]]
        else:
            res['tsallis'][j].append(Pi[0, j])
    for i in range(S):
        count += np.sum(Pi[i, :]>0)

    if lambd == lambds[0]:
        res['tsallis']['act_ratio'] = [count/(S*A)]
    else:
        res['tsallis']['act_ratio'].append(count/(S*A))
    print('Support action space is: ' + str(count/(S*A)))
    
    print('Now running expx with lambda: ' + str(lambd))
    v, Pi = expx(S, A, T0, P, R, V0, gamma, lambd)
    for i in range(S):
        p = Pi[i, :]
        p[p<=1e-4/A] = 0
        Pi[i, :] = p
    count = 0
    for j in range(A):
        if lambd == lambds[0]:
            res['expx'][j] = [Pi[0, j]]
        else:
            res['expx'][j].append(Pi[0, j])
    for i in range(S):
        count += np.sum(Pi[i, :]>0)

    if lambd == lambds[0]:
        res['expx']['act_ratio'] = [count/(S*A)]
    else:
        res['expx']['act_ratio'].append(count/(S*A))
    print('Support action space is: ' + str(count/(S*A)))

    print('Now running min with lambda: ' + str(lambd))
    v, Pi = minx(S, A, T0, P, R, V0, gamma, lambd)
    for i in range(S):
        p = Pi[i, :]
        p[p<=1e-4/A] = 0
        Pi[i, :] = p
    count = 0
    for j in range(A):
        if lambd == lambds[0]:
            res['min'][j] = [Pi[0, j]]
        else:
            res['min'][j].append(Pi[0, j])
    for i in range(S):
        count += np.sum(Pi[i, :]>0)

    if lambd == lambds[0]:
        res['min']['act_ratio'] = [count/(S*A)]
    else:
        res['min']['act_ratio'].append(count/(S*A))
    print('Support action space is: ' + str(count/(S*A)))

    print('Now running poly with lambda: ' + str(lambd))
    v, Pi = poly(S, A, T0, P, R, V0, gamma, lambd)
    for i in range(S):
        p = Pi[i, :]
        p[p<=1e-4/A] = 0
        Pi[i, :] = p
    count = 0
    for j in range(A):
        if lambd == lambds[0]:
            res['poly'][j] = [Pi[0, j]]
        else:
            res['poly'][j].append(Pi[0, j])
    for i in range(S):
        count += np.sum(Pi[i, :]>0)

    if lambd == lambds[0]:
        res['poly']['act_ratio'] = [count/(S*A)]
    else:
        res['poly']['act_ratio'].append(count/(S*A))
    print('Support action space is: ' + str(count/(S*A)))

    print('Now running mix with lambda: ' + str(lambd))
    v, Pi = mix(S, A, T0, P, R, V0, gamma, lambd)
    count = 0
    for j in range(A):
        if lambd == lambds[0]:
            res['mix'][j] = [Pi[0, j]]
        else:
            res['mix'][j].append(Pi[0, j])
    for i in range(S):
        count += np.sum(Pi[i, :]>0)

    if lambd == lambds[0]:
        res['mix']['act_ratio'] = [count/(S*A)]
    else:
        res['mix']['act_ratio'].append(count/(S*A))
    print('Support action space is: ' + str(count/(S*A)))
#json.dump(dict(zip(lambds, res)), open('cosx_action_ratio.json', 'w'))
from IPython import embed; embed()
