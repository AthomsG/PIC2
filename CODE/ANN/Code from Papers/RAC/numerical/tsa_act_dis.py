import os
import matplotlib.pyplot as plt
import numpy as np
import json
from algorithms import vi, shannon, tsallis, binary_search, cosx

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
res = []
lambds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
pis = dict()
for lambd in lambds:
    print('Now running Tsallis Entropy with lambda: ' + str(lambd))
    v, Pi = tsallis(S, A, T0, P, R, V0, gamma, lambd)
    count = 0
    for j in range(A):
        if lambd == lambds[0]:
            pis[j] = [Pi[0, j]]
        else:
            pis[j].append(Pi[0, j])
    for i in range(S):
        count += np.sum(Pi[i, :]>0)
    res.append(count/(S*A))
    print('Support action space is: ' + str(count/(S*A)))
#    print('Now running Cosx with lambda: ' + str(lambd))
#    v, Pi = cosx(S, A, T0, P, R, V0, gamma, lambd)
#    print(np.sqrt(np.sum(np.square(v[-1]-v[-2]))))
#    count = 0
#    for i in range(S):
#        count += np.sum(Pi[i, :]>0)
#    res.append(count/(S*A))
#    print('Support action space is: ' + str(count/(S*A)))
#json.dump(dict(zip(lambds, res)), open('cosx_action_ratio.json', 'w'))
for i in range(A):
    plt.plot(lambds, pis[i], label='action: '+str(i+1), linestyle='-')
plt.xlabel('Lambda', fontdict={'size': 20})
plt.ylabel('Probabilibty', fontdict={'size': 20})
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
#plt.legend(loc='lower right', prop={'size':16})
plt.title('Random MDP', fontsize=20)
fig=plt.gcf()
fig.savefig(os.path.join('./', 'tsa_act_dis.pdf'), bbox_inches='tight')
