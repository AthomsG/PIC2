import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cvxopt
cvxopt.solvers.options['show_progress'] = False
def vi(S, A, T, P, R, V0, gamma):
    vs = []
    vs.append(V0[:, 0])
    V = V0
    for i in range(1, T):
        GV = R + gamma * np.matmul(P, V)
        GV = GV.reshape(S, A)
        GV = GV.max(axis=1)
        V = GV[:, np.newaxis]
        vs.append(V[:, 0])
    return vs

def shannon(S, A, T, P, R, V0, gamma, lambd, real=False):
    vs = []
    v_real = []
    vs.append(V0[:, 0])
    V = V0
    for i in tqdm(range(1, T)):
        GV = R + gamma * np.matmul(P, V)
        GV = GV.reshape(S, A)
        GV_copy = GV.copy()
        Pi = np.exp(GV/lambd, dtype=np.float128)/np.sum(np.exp(GV/lambd, dtype=np.float128), axis=1)[:, np.newaxis]
        GV = lambd * np.log(np.sum(np.exp(GV/lambd, dtype=np.float128), axis=1))
        R_reshape = R.reshape(S, A)
        R_pi = np.sum(Pi * R_reshape, axis=1)
        Pi_axis = Pi[:, :, np.newaxis]
        P_pi = np.sum(P.reshape(S, A, S) * Pi_axis, axis=1)
        P_pi = P_pi.astype(np.float64)
        R_pi = R_pi.astype(np.float64)
        V_real = np.matmul(np.linalg.inv(np.eye(S) - gamma * P_pi), R_pi)
        v_real.append(V_real)
        V = GV[:, np.newaxis]
        #V_real = lambd * np.sum(GV_copy*np.exp(GV_copy/lambd, dtype=np.float128)/lambd, axis=1) / np.sum(np.exp(GV_copy/lambd, dtype=np.float128), axis=1)
        vs.append(GV)
    if real:
        return vs, Pi, v_real
    else:
        return vs, Pi

def binary_search(Q, lambd):
    a =  - np.max(Q/(2*lambd))
    sum_a = np.sum(np.fmax(Q/(2*lambd) + a, 0))
    if np.min(Q/(2*lambd)) >= 0:
        b = 0.01
        sum_b = np.sum(np.fmax(Q/(2*lambd) + b, 0))
        while sum_b < 1:
            b += max(np.min(Q/(2*lambd)), 0.01)
            sum_b = np.sum(np.fmax(Q/(2*lambd) + b, 0))
    elif np.min(Q/(2*lambd)) < 0:
        b = - np.min(Q/(2*lambd))
        sum_b = np.sum(np.fmax(Q/(2*lambd) + b, 0))
        while sum_b < 1:
            b += - np.min(Q/(2*lambd))
            sum_b = np.sum(np.fmax(Q/2*lambd) + b, 0)

    if sum_a < 1 and sum_b > 1:
        pass
    elif sum_a == 1:
        return a
    elif sum_b == 1:
        return b
    else:
        print('Something went wrong')
        return None

    while True:
        sum_m = np.sum(np.fmax(Q/(2*lambd) + (a+b)/2, 0))
        if np.abs(sum_m - 1) <= 1e-10:
            return (a+b)/2
        if sum_m < 1:
            a = (a + b) / 2
        elif sum_m > 1:
            b = (a + b) / 2

def cvxcos(q, lambd):
    n = q.shape[0]
    q = cvxopt.matrix(q.reshape((n, 1)))

    def F(x=None, z=None):
        if x is None: return 0, cvxopt.matrix(1.0/n, (n, 1))
        if min(x) < 0: return None
        f = - lambd * x.T * cvxopt.cos(x * np.pi/2) - x.T * q
        y = np.array(x)
        cross = y * np.sin(y * np.pi /2)
        grad = - q - lambd * cvxopt.cos(x * np.pi/2) + lambd * np.pi/2 * cvxopt.matrix(cross)
        if z is None: return f, grad.T
        h = lambd * (np.pi * x + (np.pi* np.pi /4) * x * np.cos(np.pi/2 * x))
        H = cvxopt.spdiag(z[0] * cvxopt.matrix(h))
        return f, grad.T, H
    sol = cvxopt.solvers.cp(F, cvxopt.spdiag(cvxopt.matrix(-1.0, (1, n))), cvxopt.matrix(0.0, (n, 1)), A=cvxopt.matrix(1.0, (1, n)), b=cvxopt.matrix(1.0))
    p = np.array(sol['x'])[:,0]
    return p

def cosx(S, A, T, P, R, V0, gamma, lambd, real=False):
    vs = []
    v_real = []
    vs.append(V0[:,0])
    V = V0
    for i in tqdm(range(1, T)):
        GV = R + gamma * np.matmul(P, V)
        GV = GV.reshape(S, A)
        Pi = np.array([cvxcos(q, lambd) for q in GV])
        GV = np.sum(Pi * GV + lambd * Pi * np.cos(0.5 * np.pi * Pi), axis=1)
        R_reshape = R.reshape(S, A)
        R_pi = np.sum(Pi * R_reshape, axis=1)
        Pi_axis = Pi[:, :, np.newaxis]
        P_pi = np.sum(P.reshape(S, A, S) * Pi_axis, axis=1)
        V_real = np.matmul(np.linalg.inv(np.eye(S) - gamma * P_pi), R_pi)
        v_real.append(V_real)
        V = GV[:, np.newaxis]
        vs.append(GV)
    if real:
        return vs, Pi, v_real
    else:
        return vs, Pi

def cvxexp(q, lambd):
    n = q.shape[0]
    q = cvxopt.matrix(q.reshape((n, 1)))

    def F(x=None, z=None):
        if x is None: return 0, cvxopt.matrix(float(1.0/n), (n, 1))
        if min(x) < 0: return None
        f = - lambd * x.T * (np.e - cvxopt.exp(x)) - x.T * q
        y = np.array(x)
        cross = y * np.exp(y)
        grad = - q - lambd * (np.e - cvxopt.exp(x)) + lambd * cvxopt.matrix(cross)
        if z is None: return f, grad.T
        h = (y + 2) * np.exp(y)
        H = cvxopt.spdiag(z[0] * cvxopt.matrix(h)) * lambd
        return f, grad.T, H
    sol = cvxopt.solvers.cp(F, cvxopt.spdiag(cvxopt.matrix(-1.0, (1, n))), cvxopt.matrix(0.0, (n, 1)), A=cvxopt.matrix(1.0, (1, n)), b=cvxopt.matrix(1.0))
    p = np.array(sol['x'])[:,0]
    return p

def expx(S, A, T, P, R, V0, gamma, lambd, real=False):
    vs = []
    v_real = []
    vs.append(V0[:, 0])
    V = V0
    for i in tqdm(range(1, T)):
        GV = R + gamma * np.matmul(P, V)
        GV = GV.reshape(S, A)
        Pi = np.array([cvxexp(q, lambd) for q in GV])
        GV = np.sum(Pi * GV + lambd * Pi * (np.e - np.exp(Pi)), axis=1)
        R_reshape = R.reshape(S, A)
        R_pi = np.sum(Pi * R_reshape, axis=1)
        Pi_axis = Pi[:, :, np.newaxis]
        P_pi = np.sum(P.reshape(S, A, S) * Pi_axis, axis=1)
        V_real = np.matmul(np.linalg.inv(np.eye(S) - gamma * P_pi), R_pi)
        v_real.append(V_real)
        V = GV[:, np.newaxis]
        vs.append(GV)
    if real:
        return vs, Pi, v_real
    else:
        return vs, Pi

def tsallis_cvx(q, lambd):
    n = q.shape[0]
    G = cvxopt.matrix(np.diag(-np.ones(shape=(n,))))
    h = cvxopt.matrix(np.zeros(shape=(n,)))
    A, b = cvxopt.matrix(1.0, (1,n)), cvxopt.matrix(1.0)
    Q = cvxopt.matrix(q)
    def F(x=None, z=None):
        if x is None: return 0, cvxopt.matrix(1.0, (n,1))
        if min(x) <= 0: return None
        f = - Q.T * x - lambd * x.T * (1 - x)
        grad = - Q - lambd + lambd * 2 * x
        if z is None: return f, grad.T
        H = cvxopt.spdiag(z[0] * cvxopt.matrix(2* lambd * np.ones(shape=(n,))))
        return f, grad.T, H
    sol = cvxopt.solvers.cp(F, G, h, A=A, b=b)
    p = np.array(sol['x'])[:,0]
    p[p<=1e-4/n] = 0
    return p
    
def tsallis(S, A, T, P, R, V0, gamma, lambd, real=False):
    vs = []
    v_real = []
    vs.append(V0[:,0])
    V = V0
    for i in tqdm(range(1, T)):
        GV = R + gamma * np.matmul(P, V)
        GV = GV.reshape(S, A)
        GV_copy = GV.copy()
        dual = np.array([binary_search(q, lambd) for q in GV])
        #Pi = np.array([tsallis_cvx(q, lambd) for q in GV])
        Pi = np.fmax(GV/(2*lambd)+dual[:, np.newaxis], 0)
        GV = np.sum(Pi * GV + lambd * Pi * (1 - Pi), axis=1)
        R_reshape = R.reshape(S, A)
        R_pi = np.sum(Pi * R_reshape, axis=1)
        Pi_axis = Pi[:, :, np.newaxis]
        P_pi = np.sum(P.reshape(S, A, S) * Pi_axis, axis=1)
        V_real = np.matmul(np.linalg.inv(np.eye(S) - gamma * P_pi), R_pi)
        v_real.append(V_real)
        V = GV[:, np.newaxis]
        V_real = np.sum(Pi * GV_copy, axis=1)
        vs.append(GV)
    if real:
        return vs, Pi, v_real
    else:
        return vs, Pi
        
def cvxminx(q, lambd):
    n = q.shape[0]
    q = cvxopt.matrix(q.reshape((n, 1)))
    root = 0.20318786997997992

    def F(x=None, z=None):
        if x is None: return 0, cvxopt.matrix(float(1.0/n), (n, 1))
        if min(x) < 0: return None
        f = - lambd * x.T * (cvxopt.min(-cvxopt.log(x), 2*(1 - x))) - x.T * q
        grad = np.zeros(shape=(n,1))
        grad1 = - q - lambd * (-cvxopt.log(x)) + lambd
        grad2 = - q - lambd * 2 * (1 - x) + 2 * lambd * x
        grad[np.array(x)<=root] = np.array(grad2)[np.array(x)<=root]
        grad[np.array(x)>root] = np.array(grad1)[np.array(x)>root]
        grad = cvxopt.matrix(grad)
        if z is None: return f, grad.T
        h = np.zeros(shape=(n, 1))
        h[np.array(x)<=root] = 4*lambd
        h[np.array(x)>root] = lambd / np.array(x)[np.array(x)>root]
        H = cvxopt.spdiag(z[0] * cvxopt.matrix(h))
        return f, grad.T, H
    sol = cvxopt.solvers.cp(F, cvxopt.spdiag(cvxopt.matrix(-1.0, (1, n))), cvxopt.matrix(0.0, (n, 1)), A=cvxopt.matrix(1.0, (1, n)), b=cvxopt.matrix(1.0))
    p = np.array(sol['x'])[:,0]
    return p

def minx(S, A, T, P, R, V0, gamma, lambd, real=False):
    vs = []
    v_real = []
    vs.append(V0[:, 0])
    V = V0
    for i in tqdm(range(1, T)):
        GV = R + gamma * np.matmul(P, V)
        GV = GV.reshape(S, A)
        Pi = np.array([cvxminx(q, lambd) for q in GV])
        GV = np.sum(Pi * GV + lambd * Pi * (np.minimum(-np.log(Pi), 2*(1-Pi))), axis=1)
        R_reshape = R.reshape(S, A)
        R_pi = np.sum(Pi * R_reshape, axis=1)
        Pi_axis = Pi[:, :, np.newaxis]
        P_pi = np.sum(P.reshape(S, A, S) * Pi_axis, axis=1)
        V_real = np.matmul(np.linalg.inv(np.eye(S) - gamma * P_pi), R_pi)
        v_real.append(V_real)
        V = GV[:, np.newaxis]
        vs.append(GV)
    if real:
        return vs, Pi, v_real
    else:
        return vs, Pi

def cvxpoly(q, lambd):
    n = q.shape[0]
    q = cvxopt.matrix(q.reshape((n, 1)))

    def F(x=None, z=None):
        if x is None: return 0, cvxopt.matrix(float(1.0/n), (n, 1))
        if min(x) < 0: return None
        f = - lambd * x.T * (0.5*(1-x)+(1-x**2)) - x.T * q
        grad = - q - lambd * (0.5*(1-x)+(1-x**2)) - lambd * cvxopt.mul(x, -0.5-2*x)
        if z is None: return f, grad.T
        h = lambd * (1 + 6 * x)
        H = cvxopt.spdiag(z[0] * cvxopt.matrix(h))
        return f, grad.T, H
    sol = cvxopt.solvers.cp(F, cvxopt.spdiag(cvxopt.matrix(-1.0, (1, n))), cvxopt.matrix(0.0, (n, 1)), A=cvxopt.matrix(1.0, (1, n)), b=cvxopt.matrix(1.0))
    p = np.array(sol['x'])[:,0]
    return p

def poly(S, A, T, P, R, V0, gamma, lambd, real=False):
    vs = []
    v_real = []
    vs.append(V0[:, 0])
    V = V0
    for i in tqdm(range(1, T)):
        GV = R + gamma * np.matmul(P, V)
        GV = GV.reshape(S, A)
        Pi = np.array([cvxpoly(q, lambd) for q in GV])
        GV = np.sum(Pi * GV + lambd * Pi * (0.5 * (1 - Pi) + (1 - Pi**2)), axis=1)
        R_reshape = R.reshape(S, A)
        R_pi = np.sum(Pi * R_reshape, axis=1)
        Pi_axis = Pi[:, :, np.newaxis]
        P_pi = np.sum(P.reshape(S, A, S) * Pi_axis, axis=1)
        V_real = np.matmul(np.linalg.inv(np.eye(S) - gamma * P_pi), R_pi)
        v_real.append(V_real)
        V = GV[:, np.newaxis]
        vs.append(GV)
    if real:
        return vs, Pi, v_real
    else:
        return vs, Pi

def cvxmix(q, lambd):
    n = q.shape[0]
    q = cvxopt.matrix(q.reshape((n, 1)))

    def F(x=None, z=None):
        if x is None: return 0, cvxopt.matrix(float(1.0/n), (n, 1))
        if min(x) < 0: return None
        f = - lambd * x.T * (-cvxopt.log(x)+0.5*(1-x)) - x.T * q
        grad = - q - lambd * (-cvxopt.log(x)+0.5*(1-x)) - lambd * (-1-0.5*x)
        if z is None: return f, grad.T
        h = lambd + lambd/x
        H = cvxopt.spdiag(z[0] * cvxopt.matrix(h))
        return f, grad.T, H
    sol = cvxopt.solvers.cp(F, cvxopt.spdiag(cvxopt.matrix(-1.0, (1, n))), cvxopt.matrix(0.0, (n, 1)), A=cvxopt.matrix(1.0, (1, n)), b=cvxopt.matrix(1.0))
    p = np.array(sol['x'])[:,0]
    return p

def mix(S, A, T, P, R, V0, gamma, lambd, real=False):
    vs = []
    v_real = []
    vs.append(V0[:, 0])
    V = V0
    for i in tqdm(range(1, T)):
        GV = R + gamma * np.matmul(P, V)
        GV = GV.reshape(S, A)
        Pi = np.array([cvxpoly(q, lambd) for q in GV])
        GV = np.sum(Pi * GV + lambd * Pi * (-np.log(Pi) + 0.5 * (1 - Pi)), axis=1)
        R_reshape = R.reshape(S, A)
        R_pi = np.sum(Pi * R_reshape, axis=1)
        Pi_axis = Pi[:, :, np.newaxis]
        P_pi = np.sum(P.reshape(S, A, S) * Pi_axis, axis=1)
        V_real = np.matmul(np.linalg.inv(np.eye(S) - gamma * P_pi), R_pi)
        v_real.append(V_real)
        V = GV[:, np.newaxis]
        vs.append(GV)
    if real:
        return vs, Pi, v_real
    else:
        return vs, Pi

def adaptive_shannon(S, A, T, P, R, V0, gamma, lambd0, lambd1):
    vs = []
    vs.append(V0[:, 0])
    V = V0
    for i in range(1, T):
        GV = R + gamma * np.matmul(P, V)
        GV = GV.reshape(S, A)
        lambd = np.clip(lambd0 * 0.99**(i-1), lambd1, lambd0)
        GV = lambd * np.log(np.sum(np.exp(GV/lambd, dtype=np.float128), axis=1))
        V = GV[:, np.newaxis]
        vs.append(V[:, 0])
    return vs
if __name__ == '__main__':
    np.random.seed(seed=3154)
    #S = 50
    #A = 10 
    #T0 = 5000
    #T = 500
    #gamma = 0.99

    #sparse = np.random.binomial(1, 0.05, size=(A*S, S))
    #for i in range(A*S):
    #    if sum(sparse[i, :]) == 0:
    #        sparse[i, np.random.randint(S)] = 1
    #P = np.random.uniform(0, 1, size=(A * S, S)) 
    #P = P / np.sum(P, 1)[:, np.newaxis]
    #R = np.random.uniform(low=0, high=1, size=(A*S, 1))
    #V0 = np.random.uniform(low=0, high=0.1, size=(S, 1))/(1-gamma)
    S = N = 100                                                                                                                                                                                                        
    A = 2        
    T0 = 5000
    T = 500                                                                                                                                                                                                       
    gamma = 0.99                                                                                                                                                                                                   
                                                                                                                                                                                                                   
    P = np.zeros(shape=(N, A, N))                                                                                                                                                                                  
    for i in range(N):                                                                                                                                                                                             
        if i == 0:                                                                                                                                                                                                 
            P[0, 1, 1] = 1                                                                                                                                                                                         
            P[0, 0, 0] = 1                                                                                                                                                                                         
        elif i == N-1:                                                                                                                                                                                             
            P[i, 1, i] = 1                                                                                                                                                                                         
            P[i, 0, i] = 1                                                                                                                                                                                         
        else:                                                                                                                                                                                                      
            P[i, 1, i+1] = 0.9                                                                                                                                                                                     
            P[i, 0, i+1] = 0.1                                                                                                                                                                                     
            P[i, 1, i-1] = 0.1                                                                                                                                                                                     
            P[i, 0, i-1] = 0.9    
    P = P.reshape(N*A, N)
    R = np.zeros(shape=(N, A))
    R[0, 0] = 0.1                                                                                                                                                                                                  
    R[0, 1] = 0.1                                                                                                                                                                                                  
    R[N-1, 1] = 1                                                                                                                                                                                                  
    R[N-1, 0] = 1             
    R = R.reshape(N*A, 1)
    V0 = np.random.uniform(low=0, high=1, size=(S, 1))/(1-gamma)          

    true_values = vi(S, A, T0, P, R, V0, gamma)
    true_value = true_values[-1]
    Names = {'vi': 'Value Iteration'}
    ERR = {}
    vi_res = vi(S, A, T, P, R, V0, gamma)
    vi_err = np.abs(vi_res - true_value).max(axis=1)
    ERR['vi'] = vi_err
    for lambd in [0.1]:
        print('Now running Shannon with lambda:' +str(lambd))
        shannon_res, Pi = shannon(S, A, T, P, R, V0, gamma, lambd)
        shannon_err = np.abs(shannon_res - true_value).max(axis=1)
        ERR['shannon-'+str(lambd)] = shannon_err
        Names['shannon-'+str(lambd)] = 'Shannon Entropy-'+str(lambd)
        print('Now running Tsallis with lambda:' +str(lambd))
        tsallis_res, Pi = tsallis(S, A, T, P, R, V0, gamma, lambd)
        tsallis_err = np.abs(tsallis_res - true_value).max(axis=1)
        print(tsallis_err[-1])
        ERR['tsallis-'+str(lambd)] = tsallis_err
        Names['tsallis-'+str(lambd)] = 'Tsallis Entropy-'+str(lambd)
        print('Now running cosx with lambda:' +str(lambd))
        cosx_res, Pi = cosx(S, A, T, P, R, V0, gamma, lambd)
        cosx_err = np.abs(cosx_res - true_value).max(axis=1)
        ERR['cosx-'+str(lambd)] = cosx_err
        Names['cosx-'+str(lambd)] = 'Cosx Entropy-'+str(lambd)
        print('Now running expx with lambda:' +str(lambd))
        expx_res, Pi = expx(S, A, T, P, R, V0, gamma, lambd)
        expx_err = np.abs(expx_res - true_value).max(axis=1)
        ERR['expx-'+str(lambd)] = expx_err
        Names['expx-'+str(lambd)] = 'Expx Entropy-'+str(lambd)
        count = 0
        for i in range(S):
            count += np.sum(Pi[i,:]>1e-4/A)
        print(count)
    #adaptive_true_values  = adaptive_shannon(S, A, T0, P, R, V0, gamma, 0.1, 0.001)
    #adaptive_true_value = adaptive_true_values[-1]
    #V0_lambd = V0 + 0.1 * np.log(A) / (1-gamma)
    #adaptive_shannon_res = adaptive_shannon(S, A, T, P, R, V0, gamma, 0.05, 0.01)
    #adaptive_shannon_err = np.abs(adaptive_shannon_res - true_value).max(axis=1)
    #ERR['adaptive shannon'] = adaptive_shannon_err
    #Names['adaptive shannon'] = 'Adaptive Shannon'


    Line = '-'
    for key in Names.keys():
        plt.plot(list(range(T)), ERR[key], label=Names[key], linestyle=Line)
    plt.xlabel('Iterations', fontdict={'size': 20})
    plt.ylabel('Errors', fontdict={'size': 20})
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(loc='upper right', prop={'size': 16})
    plt.title('Numerical Results', fontsize=20)
    fig = plt.gcf()
    fig.savefig(os.path.join('./', 'results.pdf'), bbox_inches = 'tight')
