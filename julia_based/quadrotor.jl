MIN_C1, MAX_C1 = -15., 15.
MIN_C, MAX_C = -1., 1.
MIN_V, MAX_V = -1., 1.
MASS_INV = 1.
BETA = 1.
EPS = 2.107342e-08

function _compute_derivatives(q, u)
    qdot = zeros(13)
    qdot[1:3] = q[8:10]
    qomega = zeros(4) #[ x, y, z, w,]
    qomega[1:3] = 0.5 .*q[11:13]
    qomega = enforce_bounds_quaternion(qomega)
    delta = q[4] * qomega[1] + q[5] * qomega[2] + q[6] * qomega[3]
    qdot[4:7] = qomega - delta .* q[4:7]
    qdot[8] = MASS_INV * (-2 * u[1]*(q[7]*q[5]+q[4]*q[6]) - BETA * q[8])
    qdot[9] = MASS_INV * (-2 * u[1]*(q[5]*q[6]-q[7]*q[4]) - BETA * q[9])
    qdot[10] = MASS_INV * (- u[1] * (q[7] * q[7] - q[4] * q[4] - q[5] * q[5] +q[6]*q[6]) - BETA*q[10]) - 9.81
    qdot[11:13] = u[2:4]
    qdot
end

function enforce_bounds_quaternion(pose)
    # enforce quaternion
    # http://stackoverflow.com/questions/11667783/quaternion-and-normalization/12934750#12934750
    nrmsq = sum(pose .^ 2)
    if abs(1.0 - nrmsq) < EPS
        pose .*= 2.0 / (1.0 + nrmsq)
    elseif nrmsq < 1e-6
        pose = [0, 0, 0, 1] 
    else
        pose .*= 1.0 / sqrt(nrmsq)
    end
    pose
end

function propagate(start_state, control, num_steps, integration_step)
    q = start_state
    control[1] = clamp(control[1], MIN_C1, MAX_C1)
    control[2] = clamp(control[2], MIN_C, MAX_C)
    control[3] = clamp(control[3], MIN_C, MAX_C)
    control[4] = clamp(control[4], MIN_C, MAX_C)
    for i =1:num_steps
        q += integration_step * _compute_derivatives(q, control)
        q[8:13] = clamp.(q[8:13], MIN_V, MAX_V)
        q[4:7] = enforce_bounds_quaternion(q[4:7])
    end

    q
end

# start = [-4.96396, -4.63075, 3.70707, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0] 
# u = [1,1,1,1]
# println(propagate(start, u, 0.02/0.002, 0.002))