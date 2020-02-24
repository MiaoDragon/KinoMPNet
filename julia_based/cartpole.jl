Inertia = 10
L = 2.5
M = 10
m = 5
g = 9.8
#  Height of the cart
H = 0.5

STATE_X = 1
STATE_V = 2
STATE_THETA = 3
STATE_W = 4

MIN_X = -30
MAX_X = 30
MIN_V = -40
MAX_V = 40
MIN_W = -2
MAX_W = 2


function enforce_bounds(temp_state)
    if temp_state[1] < MIN_X
        temp_state[1] = MIN_X
    elseif temp_state[1] > MAX_X
        temp_state[1] = MAX_X
    end
    if temp_state[2] < MIN_V
        temp_state[2] = MIN_V
    elseif temp_state[2] > MAX_V
        temp_state[2] = MAX_V
    end
    
    if temp_state[3] < -pi
        temp_state[3] += 2 * pi
    elseif temp_state[3] > pi
        temp_state[3] -= 2 * pi
    end
    if temp_state[4] < MIN_W
        temp_state[4] = MIN_W
    elseif temp_state[4] > MAX_W
        temp_state[4] = MAX_W
    end
    temp_state
end




function update_derivative(state, control)
    #  Height of the cart
    
    deriv = deepcopy(state)
    temp_state = deepcopy(state)
    _v = temp_state[STATE_V]
    _w = temp_state[STATE_W]
    _theta = temp_state[STATE_THETA]
    _a = control
    mass_term = (M + m)*(Inertia + m * L * L) - m * m * L * L * cos(_theta) * cos(_theta)
    deriv[STATE_X] = _v
    deriv[STATE_THETA] = _w
    mass_term = (1.0 / mass_term)
    deriv[STATE_V] = ((Inertia + m * L * L)*(_a + m * L * _w * _w * sin(_theta)) + m * m * L * L * cos(_theta) * sin(_theta) * g) * mass_term
    deriv[STATE_W] = ((-m * L * cos(_theta))*(_a + m * L * _w * _w * sin(_theta))+(M + m)*(-m * g * L * sin(_theta))) * mass_term
    deriv
end

function propagate(start_state, control, num_steps, integration_step)
    temp_state = start_state
    for _ in 1:num_steps
        deriv = update_derivative(temp_state, control)
        temp_state[1] += integration_step * deriv[1]
        temp_state[2] += integration_step * deriv[2]
        temp_state[3] += integration_step * deriv[3]
        temp_state[4] += integration_step * deriv[4]
        temp_state = enforce_bounds(temp_state)
    end
    temp_state
end