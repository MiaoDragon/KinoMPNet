import torch
import jax
import numpy as np
import sys
sys.path.append('..')
from plan_utility.line_line_cc import line_line_cc

def normalize(x, bound):
    # normalize to -1 ~ 1  (bound can be a tensor)
    #return x
    bound = torch.tensor(bound)
    if len(x.size()) > 1:
        if len(x[0]) != len(bound):
            x[:,:-len(bound)] = x[:,:-len(bound)] / bound
            x[:,-len(bound):] = x[:,-len(bound):] / bound
        else:
            x = x / bound
    else:
        if len(x) != len(bound):
            x[:-len(bound)] = x[:-len(bound)] / bound
            x[-len(bound):] = x[-len(bound):] / bound
        else:
            x = x / bound
    return x
def unnormalize(x, bound):
    # normalize to -1 ~ 1  (bound can be a tensor)
    # x only one dim
    #return x
    bound = torch.tensor(bound)
    if len(x.size()) > 1:
        if len(x[0]) != len(bound):
            x[:,:-len(bound)] = x[:,:-len(bound)] * bound
            x[:,-len(bound):] = x[:,-len(bound):] * bound
        else:
            x = x * bound
    else:
        if len(x) != len(bound):
            x[:-len(bound)] = x[:-len(bound)] * bound
            x[-len(bound):] = x[-len(bound):] * bound
        else:
            x = x * bound
    return x


def dynamics(state, control):
    '''
    implement the function x_dot = f(x,u)
    return the derivative w.r.t. x
    '''
    I = 10
    L = 2.5
    M = 10
    m = 5
    g = 9.8
    H = 0.5  # cart
    # define the name for each state index and action index
    STATE_X, STATE_V, STATE_THETA, STATE_W = 0, 1, 2, 3
    CONTROL_A = 0
    # define boundary
    MIN_X = -30
    MAX_X = 30
    MIN_V = -40
    MAX_V = 40
    MIN_W = -2
    MAX_W = 2
    # obstacle information
    OBS_W = 4
    _v = state[STATE_V]
    _w = state[STATE_W]
    _theta = state[STATE_THETA]
    _a = control[CONTROL_A]
    mass_term = (M + m)*(I + m * L * L) - \
            m * m * L * L * np.cos(_theta) * np.cos(_theta)

    deriv = np.zeros(4)  # init derivative
    deriv[STATE_X] = _v
    deriv[STATE_THETA] = _w
    mass_term = (1.0 / mass_term)
    # normalize: added (1/max_X) term
    #deriv[STATE_V] = (1/MAX_X)*((I + m * L * L)* \
    deriv[STATE_V] = ((I + m * L * L)* \
        (_a + m * L * _w * _w * np.sin(_theta)) + \
        m * m * L * L * np.cos(_theta) * np.sin(_theta) * g) * mass_term
    deriv[STATE_W] = ((-m * L * np.cos(_theta)) * \
        (_a + m * L * _w * _w * np.sin(_theta))+(M + m) * \
        (-m * g * L * np.sin(_theta))) * mass_term
    return deriv

def enforce_bounds(state):
    '''

    check if state satisfies the bound
    apply threshold to velocity and angle
    return a new state toward which the bound has been enforced
    '''
    I = 10
    L = 2.5
    M = 10
    m = 5
    g = 9.8
    H = 0.5  # cart
    # define the name for each state index and action index
    STATE_X, STATE_V, STATE_THETA, STATE_W = 0, 1, 2, 3
    CONTROL_A = 0
    # define boundary
    MIN_X = -30
    MAX_X = 30
    MIN_V = -40
    MAX_V = 40
    MIN_W = -2
    MAX_W = 2
    # obstacle information
    OBS_W = 4
    new_state = np.array(state)
    """
    if state[STATE_V] < MIN_V/30.:
        new_state[STATE_V] = MIN_V/30.
    elif state[STATE_V] > MAX_V/30.:
        new_state[STATE_V] = MAX_V/30.
    """
    if(state[STATE_X]<MIN_X):
        new_state[STATE_X]=MIN_X
    elif(state[STATE_X]>MAX_X):
        new_state[STATE_X]=MAX_X

    if state[STATE_V] < MIN_V:
        new_state[STATE_V] = MIN_V
    elif state[STATE_V] > MAX_V:
        new_state[STATE_V] = MAX_V

    if state[STATE_THETA] < -np.pi:
        new_state[STATE_THETA] += 2*np.pi
    elif state[STATE_THETA] > np.pi:
        new_state[STATE_THETA] -= 2*np.pi

    if state[STATE_W] < MIN_W:
        new_state[STATE_W] = MIN_W
    elif state[STATE_W] > MAX_W:
        new_state[STATE_W] = MAX_W
    return new_state



def jax_dynamics(state, control):
    '''
    implement the function x_dot = f(x,u)
    return the derivative w.r.t. x
    '''
    I = 10
    L = 2.5
    M = 10
    m = 5
    g = 9.8
    H = 0.5  # cart
    # define the name for each state index and action index
    STATE_X, STATE_V, STATE_THETA, STATE_W = 0, 1, 2, 3
    CONTROL_A = 0
    # define boundary
    MIN_X = -30
    MAX_X = 30
    MIN_V = -40
    MAX_V = 40
    MIN_W = -2
    MAX_W = 2
    # obstacle information
    OBS_W = 4
    _v = state[STATE_V]
    _w = state[STATE_W]
    _theta = state[STATE_THETA]
    _a = control[CONTROL_A]
    mass_term = (M + m)*(I + m * L * L) - \
            m * m * L * L * jax.numpy.cos(_theta) * jax.numpy.cos(_theta)

    deriv = [0,0,0,0]  # init derivative
    deriv[STATE_X] = _v
    deriv[STATE_THETA] = _w
    mass_term = (1.0 / mass_term)
    # normalize (added 1/max_X term)
    #deriv[STATE_V] = (1/MAX_X)*((I + m * L * L)* \
    deriv[STATE_V] = ((I + m * L * L)* \
        (_a + m * L * _w * _w * jax.numpy.sin(_theta)) + \
        m * m * L * L * jax.numpy.cos(_theta) * jax.numpy.sin(_theta) * g) * mass_term
    deriv[STATE_W] = ((-m * L * jax.numpy.cos(_theta)) * \
        (_a + m * L * _w * _w * jax.numpy.sin(_theta))+(M + m) * \
        (-m * g * L * jax.numpy.sin(_theta))) * mass_term
    return jax.numpy.asarray(deriv)


def IsInCollision(x, obc, obc_width=4.):
    I = 10
    L = 2.5
    M = 10
    m = 5
    g = 9.8
    H = 0.5

    STATE_X = 0
    STATE_V = 1
    STATE_THETA = 2
    STATE_W = 3
    CONTROL_A = 0

    MIN_X = -30
    MAX_X = 30
    MIN_V = -40
    MAX_V = 40
    MIN_W = -2
    MAX_W = 2

    
    if x[0] < MIN_X or x[0] > MAX_X:
        return True
    
    H = 0.5
    pole_x1 = x[0]
    pole_y1 = H
    pole_x2 = x[0] + L * np.sin(x[2])
    pole_y2 = H + L * np.cos(x[2])

    
    for i in range(len(obc)):
        for j in range(0, 8, 2):
            x1 = obc[i][j]
            y1 = obc[i][j+1]
            x2 = obc[i][(j+2) % 8]
            y2 = obc[i][(j+3) % 8]
            if line_line_cc(pole_x1, pole_y1, pole_x2, pole_y2, x1, y1, x2, y2):
                return True
    return False