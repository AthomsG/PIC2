from .dqn import Dqn
from .rqn import Rqn

def get_estimator(model_name, n_ac, lr, discount, reg='none', lambd=0,  **kwargs):
    if model_name == 'dqn':
        estimator = Dqn(n_ac, lr=lr, discount=discount)
    elif model_name == 'rqn':
        estimator = Rqn(n_ac, lr=lr, discount=discount, reg=reg, lambd=lambd)
    else:
        raise Exception('{} is not supported!'.format(model_name))

    return estimator
