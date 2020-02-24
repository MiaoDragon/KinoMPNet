STATE_THETA_1, STATE_THETA_2, STATE_V_1, STATE_V_2 = 1,2,3,4
MIN_V_1, MAX_V_1 = -6., 6.
MIN_V_2, MAX_V_2 = -6., 6.
MIN_TORQUE, MAX_TORQUE = -4., 4.

MIN_ANGLE, MAX_ANGLE = -pi, pi

LENGTH = 20.
m = 1.0
lc = 0.5
lc2 = 0.25
l2 = 1.
I1 = 0.2
I2 = 1.0
l = 1.0
g = 9.81



function _compute_derivatives(state, control)
    theta2 = state[STATE_THETA_2]
    theta1 = state[STATE_THETA_1] - pi/2
    theta1dot = state[STATE_V_1]
    theta2dot = state[STATE_V_2]
    _tau = control
    

    d11 = m * lc2 + m * (l2 + lc2 + 2 * l * lc * cos(theta2)) + I1 + I2
    d22 = m * lc2 + I2
    d12 = m * (lc2 + l * lc * cos(theta2)) + I2
    d21 = d12

    c1 = -m * l * lc * theta2dot * theta2dot * sin(theta2) - (2 * m * l * lc * theta1dot * theta2dot * sin(theta2))
    c2 = m * l * lc * theta1dot * theta1dot * sin(theta2)
    g1 = (m * lc + m * l) * g * cos(theta1) + (m * lc * g * cos(theta1 + theta2))
    g2 = m * lc * g * cos(theta1 + theta2)

    deriv = copy(state)
    deriv[STATE_THETA_1] = theta1dot
    deriv[STATE_THETA_2] = theta2dot

    u2 = _tau - 1 * .1 * theta2dot
    u1 = -1 * .1 * theta1dot
    theta1dot_dot = (d22 * (u1 - c1 - g1) - d12 * (u2 - c2 - g2)) / (d11 * d22 - d12 * d21)
    theta2dot_dot = (d11 * (u2 - c2 - g2) - d21 * (u1 - c1 - g1)) / (d11 * d22 - d12 * d21)
    deriv[STATE_V_1] = theta1dot_dot
    deriv[STATE_V_2] = theta2dot_dot
    deriv
end
function propagate(start_state, control, num_steps, integration_step)
    state = start_state
    for i =1:num_steps
        state += integration_step * _compute_derivatives(state, control)

        if state[1] < -pi
            state[1] += 2*pi
        elseif state[1] > pi
            state[1] -= 2 * pi
        end
        if state[2] < -pi
            state[2] += 2*pi
        elseif state[2] > pi
            state[2] -= 2 * pi
        end
        state[3] = clamp(state[3], MIN_V_1, MAX_V_1)
        state[4] = clamp(state[4], MIN_V_2, MAX_V_2)
    end
    state
end
# println(propagate([ 0.24618453, -0.5358435 ,  1.08646313, -2.47822479], -1.71350494, convert(Int,round(0.18/0.002)), 0.002))

