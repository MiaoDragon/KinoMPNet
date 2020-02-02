import torch
import numpy as np
import sys
sys.path.append('..')
import jax
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
    STATE_THETA_1, STATE_THETA_2, STATE_V_1, STATE_V_2 = 0, 1, 2, 3
    MIN_V_1, MAX_V_1 = -6., 6.
    MIN_V_2, MAX_V_2 = -6., 6.
    MIN_TORQUE, MAX_TORQUE = -4., 4.

    MIN_ANGLE, MAX_ANGLE = -np.pi, np.pi

    LENGTH = 20.
    m = 1.0
    lc = 0.5
    lc2 = 0.25
    l2 = 1.
    I1 = 0.2
    I2 = 1.0
    l = 1.0
    g = 9.81

    theta2 = state[STATE_THETA_2]
    theta1 = state[STATE_THETA_1] - np.pi/2
    theta1dot = state[STATE_V_1]
    theta2dot = state[STATE_V_2]
    _tau = control[0]
    m = m
    l2 = l2
    lc2 = lc2
    l = l
    lc = lc
    I1 = I1
    I2 = I2

    d11 = m * lc2 + m * (l2 + lc2 + 2 * l * lc * np.cos(theta2)) + I1 + I2
    d22 = m * lc2 + I2
    d12 = m * (lc2 + l * lc * np.cos(theta2)) + I2
    d21 = d12

    c1 = -m * l * lc * theta2dot * theta2dot * np.sin(theta2) - (2 * m * l * lc * theta1dot * theta2dot * np.sin(theta2))
    c2 = m * l * lc * theta1dot * theta1dot * np.sin(theta2)
    g1 = (m * lc + m * l) * g * np.cos(theta1) + (m * lc * g * np.cos(theta1 + theta2))
    g2 = m * lc * g * np.cos(theta1 + theta2)

    deriv = state.copy()
    deriv[STATE_THETA_1] = theta1dot
    deriv[STATE_THETA_2] = theta2dot

    u2 = _tau - 1 * .1 * theta2dot
    u1 = -1 * .1 * theta1dot
    theta1dot_dot = (d22 * (u1 - c1 - g1) - d12 * (u2 - c2 - g2)) / (d11 * d22 - d12 * d21)
    theta2dot_dot = (d11 * (u2 - c2 - g2) - d21 * (u1 - c1 - g1)) / (d11 * d22 - d12 * d21)
    deriv[STATE_V_1] = theta1dot_dot
    deriv[STATE_V_2] = theta2dot_dot
    return deriv

def enforce_bounds(state):
    STATE_THETA_1, STATE_THETA_2, STATE_V_1, STATE_V_2 = 0, 1, 2, 3
    MIN_V_1, MAX_V_1 = -6., 6.
    MIN_V_2, MAX_V_2 = -6., 6.
    MIN_TORQUE, MAX_TORQUE = -4., 4.

    MIN_ANGLE, MAX_ANGLE = -np.pi, np.pi
    state = np.array(state)
    if state[0] < -np.pi:
        state[0] += 2*np.pi
    elif state[0] > np.pi:
        state[0] -= 2 * np.pi
    if state[1] < -np.pi:
        state[1] += 2*np.pi
    elif state[1] > np.pi:
        state[1] -= 2 * np.pi

    state[2:] = np.clip(
        state[2:],
        [MIN_V_1, MIN_V_2],
        [MAX_V_1, MAX_V_2])
    return state

def jax_dynamics(state, control):
    STATE_THETA_1, STATE_THETA_2, STATE_V_1, STATE_V_2 = 0, 1, 2, 3
    MIN_V_1, MAX_V_1 = -6., 6.
    MIN_V_2, MAX_V_2 = -6., 6.
    MIN_TORQUE, MAX_TORQUE = -4., 4.

    MIN_ANGLE, MAX_ANGLE = -np.pi, np.pi

    LENGTH = 20.
    m = 1.0
    lc = 0.5
    lc2 = 0.25
    l2 = 1.
    I1 = 0.2
    I2 = 1.0
    l = 1.0
    g = 9.81

    theta2 = state[STATE_THETA_2]
    theta1 = state[STATE_THETA_1] - np.pi/2
    theta1dot = state[STATE_V_1]
    theta2dot = state[STATE_V_2]
    _tau = control[0]
    m = m
    l2 = l2
    lc2 = lc2
    l = l
    lc = lc
    I1 = I1
    I2 = I2

    d11 = m * lc2 + m * (l2 + lc2 + 2 * l * lc * jax.numpy.cos(theta2)) + I1 + I2
    d22 = m * lc2 + I2
    d12 = m * (lc2 + l * lc * jax.numpy.cos(theta2)) + I2
    d21 = d12

    c1 = -m * l * lc * theta2dot * theta2dot * jax.numpy.sin(theta2) - (2 * m * l * lc * theta1dot * theta2dot * jax.numpy.sin(theta2))
    c2 = m * l * lc * theta1dot * theta1dot * jax.numpy.sin(theta2)
    g1 = (m * lc + m * l) * g * jax.numpy.cos(theta1) + (m * lc * g * jax.numpy.cos(theta1 + theta2))
    g2 = m * lc * g * jax.numpy.cos(theta1 + theta2)

    deriv = [0,0,0,0]
    deriv[STATE_THETA_1] = theta1dot
    deriv[STATE_THETA_2] = theta2dot

    u2 = _tau - 1 * .1 * theta2dot
    u1 = -1 * .1 * theta1dot
    theta1dot_dot = (d22 * (u1 - c1 - g1) - d12 * (u2 - c2 - g2)) / (d11 * d22 - d12 * d21)
    theta2dot_dot = (d11 * (u2 - c2 - g2) - d21 * (u1 - c1 - g1)) / (d11 * d22 - d12 * d21)
    deriv[STATE_V_1] = theta1dot_dot
    deriv[STATE_V_2] = theta2dot_dot
    return jax.numpy.asarray(deriv)

def IsInCollision(x, obc, obc_width=6.):
    STATE_THETA_1, STATE_THETA_2, STATE_V_1, STATE_V_2 = 0, 1, 2, 3
    MIN_V_1, MAX_V_1 = -6., 6.
    MIN_V_2, MAX_V_2 = -6., 6.
    MIN_TORQUE, MAX_TORQUE = -4., 4.

    MIN_ANGLE, MAX_ANGLE = -np.pi, np.pi

    LENGTH = 20.
    m = 1.0
    lc = 0.5
    lc2 = 0.25
    l2 = 1.
    I1 = 0.2
    I2 = 1.0
    l = 1.0
    g = 9.81
    pole_x0 = 0.
    pole_y0 = 0.
    pole_x1 = LENGTH * np.cos(x[STATE_THETA_1] - np.pi / 2)
    pole_y1 = LENGTH * np.sin(x[STATE_THETA_1] - np.pi / 2)
    pole_x2 = pole_x1 + LENGTH * np.cos(x[STATE_THETA_1] + x[STATE_THETA_2] - np.pi / 2)
    pole_y2 = pole_y1 + LENGTH * np.sin(x[STATE_THETA_1] + x[STATE_THETA_2] - np.pi / 2)
    for i in range(len(obc)):
        for j in range(0, 8, 2):
            x1 = obc[i][j]
            y1 = obc[i][j+1]
            x2 = obc[i][(j+2) % 8]
            y2 = obc[i][(j+3) % 8]
            if line_line_cc(pole_x0, pole_y0, pole_x1, pole_y1, x1, y1, x2, y2):
                return True
            if line_line_cc(pole_x1, pole_y1, pole_x2, pole_y2, x1, y1, x2, y2):
                return True
    return False