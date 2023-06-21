import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cvxopt
cvxopt.solvers.options['show_progress'] = False
cvxopt.solvers.options['maxiters'] = 10

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

def cvxtsallis(q, lambd):
    n = q.shape[0]
    G = cvxopt.matrix(np.diag(-np.ones(shape=(n,))))
    h = cvxopt.matrix(np.zeros(shape=(n,)))
    A, b = cvxopt.matrix(1.0, (1,n)), cvxopt.matrix(1.0)
    Q = cvxopt.matrix(q.reshape(n, 1))
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
        h[np.array(x)>root] = lambd / (np.array(x)[np.array(x)>root])
        H = cvxopt.spdiag(z[0] * cvxopt.matrix(h))
        return f, grad.T, H
    sol = cvxopt.solvers.cp(F, cvxopt.spdiag(cvxopt.matrix(-1.0, (1, n))), cvxopt.matrix(0.0, (n, 1)), A=cvxopt.matrix(1.0, (1, n)), b=cvxopt.matrix(1.0))
    p = np.array(sol['x'])[:,0]
    return p

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
