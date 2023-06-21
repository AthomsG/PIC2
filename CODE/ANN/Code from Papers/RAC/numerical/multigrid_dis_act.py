import os
import matplotlib.pyplot as plt
import numpy as np
import json
from algorithms import vi, shannon, tsallis, binary_search, cosx, expx
np.random.seed(seed=3154)

N = 20
S = (2 * N - 1)*(2 * N - 1)
A = 4
T0 = 500
gamma = 0.99

maps = dict()
index = 0
for i in range(-N+1, N):
    for j in range(-N+1, N):
        maps[(i,j)] = index
        index += 1
P = np.zeros(shape=(S, A, S))

for pos in maps.keys():
    i, j = pos
    if abs(i) != N-1 and abs(j) != N-1:
        P[maps[(i, j)], 0, maps[(i+1, j)]] = 1
        P[maps[(i, j)], 1, maps[(i-1, j)]] = 1
        P[maps[(i, j)], 2, maps[(i, j+1)]] = 1
        P[maps[(i, j)], 3, maps[(i, j-1)]] = 1
    elif i == N-1 and abs(j) != N-1:
        P[maps[(i, j)], 0, maps[(i, j)]] = 1
        P[maps[(i, j)], 1, maps[(i-1, j)]] = 1
        P[maps[(i, j)], 2, maps[(i, j+1)]] = 1
        P[maps[(i, j)], 3, maps[(i, j-1)]] = 1
    elif i == -N+1 and abs(j) != N-1:
        P[maps[(i, j)], 0, maps[(i+1, j)]] = 1
        P[maps[(i, j)], 1, maps[(i, j)]] = 1
        P[maps[(i, j)], 2, maps[(i, j+1)]] = 1
        P[maps[(i, j)], 3, maps[(i, j-1)]] = 1
    elif j == N-1 and abs(i) != N-1:
        P[maps[(i, j)], 0, maps[(i+1, j)]] = 1
        P[maps[(i, j)], 1, maps[(i-1, j)]] = 1
        P[maps[(i, j)], 2, maps[(i, j)]] = 1
        P[maps[(i, j)], 3, maps[(i, j-1)]] = 1
    elif j == -N+1 and abs(i) != N-1:
        P[maps[(i, j)], 0, maps[(i+1, j)]] = 1
        P[maps[(i, j)], 1, maps[(i-1, j)]] = 1
        P[maps[(i, j)], 2, maps[(i, j+1)]] = 1
        P[maps[(i, j)], 3, maps[(i, j)]] = 1
    elif i == -N+1 and j == -N+1:
        P[maps[(i, j)], 0, maps[(i+1, j)]] = 1
        P[maps[(i, j)], 1, maps[(i, j)]] = 1
        P[maps[(i, j)], 2, maps[(i, j+1)]] = 1
        P[maps[(i, j)], 3, maps[(i, j)]] = 1
    elif i == N-1 and j == -N+1:
        P[maps[(i, j)], 0, maps[(i, j)]] = 1
        P[maps[(i, j)], 1, maps[(i-1, j)]] = 1
        P[maps[(i, j)], 2, maps[(i, j+1)]] = 1
        P[maps[(i, j)], 3, maps[(i, j)]] = 1
    elif i == -N+1 and j == N-1:
        P[maps[(i, j)], 0, maps[(i+1, j)]] = 1
        P[maps[(i, j)], 1, maps[(i, j)]] = 1
        P[maps[(i, j)], 2, maps[(i, j)]] = 1
        P[maps[(i, j)], 3, maps[(i, j-1)]] = 1
    elif i == N-1 and j == N-1:
        P[maps[(i, j)], 0, maps[(i, j)]] = 1
        P[maps[(i, j)], 1, maps[(i-1, j)]] = 1
        P[maps[(i, j)], 2, maps[(i, j)]] = 1
        P[maps[(i, j)], 3, maps[(i, j-1)]] = 1

P = P.reshape(S*A, S)
R = np.zeros(shape=(S, A))
R[maps[(N-1, N-1)],] = 1
R[maps[(N-1, -N+1)],] = 1
R[maps[(-N+1, N-1)],] = 1
R[maps[(-N+1, -N+1)],] = 1
R = R.reshape(S*A, 1)
V0 = np.random.uniform(low=0, high=1, size=(S, 1))/(1-gamma)

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
        'logx': {'lambdas': lambds}}
for lambd in lambds:
    print('Now running Logx with lambda: ' + str(lambd))
    v, Pi = shannon(S, A, T0, P, R, V0, gamma, lambd)
    count = 0
    for j in range(A):
        if lambd == lambds[0]:
            res['logx']['(0,0):'+str(j)] = [Pi[maps[(0, 0)], j]]
            res['logx']['(N/2, N/2):'+str(j)] = [Pi[maps[(int(N/2), int(N/2))], j]]
            res['logx']['(0, N/2):'+str(j)] = [Pi[maps[(0, int(N/2))], j]]
        else:
            res['logx']['(0,0):'+str(j)].append(Pi[maps[(0, 0)], j])
            res['logx']['(N/2, N/2):'+str(j)].append(Pi[maps[(int(N/2), int(N/2))], j])
            res['logx']['(0, N/2):'+str(j)].append(Pi[maps[(0, int(N/2))], j])
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
            res['cosx']['(0,0):'+str(j)] = [Pi[maps[(0, 0)], j]]
            res['cosx']['(N/2, N/2):'+str(j)] = [Pi[maps[(int(N/2), int(N/2))], j]]
            res['cosx']['(0, N/2):'+str(j)] = [Pi[maps[(0, int(N/2))], j]]
        else:
            res['cosx']['(0,0):'+str(j)].append(Pi[maps[(0, 0)], j])
            res['cosx']['(N/2, N/2):'+str(j)].append(Pi[maps[(int(N/2), int(N/2))], j])
            res['cosx']['(0, N/2):'+str(j)].append(Pi[maps[(0, int(N/2))], j])
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
            res['tsallis']['(0,0):'+str(j)] = [Pi[maps[(0, 0)], j]]
            res['tsallis']['(N/2, N/2):'+str(j)] = [Pi[maps[(int(N/2), int(N/2))], j]]
            res['tsallis']['(0, N/2):'+str(j)] = [Pi[maps[(0, int(N/2))], j]]
        else:
            res['tsallis']['(0,0):'+str(j)].append(Pi[maps[(0, 0)], j])
            res['tsallis']['(N/2, N/2):'+str(j)].append(Pi[maps[(int(N/2), int(N/2))], j])
            res['tsallis']['(0, N/2):'+str(j)].append(Pi[maps[(0, int(N/2))], j])
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
            res['expx']['(0,0):'+str(j)] = [Pi[maps[(0, 0)], j]]
            res['expx']['(N/2, N/2):'+str(j)] = [Pi[maps[(int(N/2), int(N/2))], j]]
            res['expx']['(0, N/2):'+str(j)] = [Pi[maps[(0, int(N/2))], j]]
        else:
            res['expx']['(0,0):'+str(j)].append(Pi[maps[(0, 0)], j])
            res['expx']['(N/2, N/2):'+str(j)].append(Pi[maps[(int(N/2), int(N/2))], j])
            res['expx']['(0, N/2):'+str(j)].append(Pi[maps[(0, int(N/2))], j])
    for i in range(S):
        count += np.sum(Pi[i, :]>0)

    if lambd == lambds[0]:
        res['expx']['act_ratio'] = [count/(S*A)]
    else:
        res['expx']['act_ratio'].append(count/(S*A))
    print('Support action space is: ' + str(count/(S*A)))

#json.dump(dict(zip(lambds, res)), open('cosx_action_ratio.json', 'w'))
from IPython import embed; embed()
