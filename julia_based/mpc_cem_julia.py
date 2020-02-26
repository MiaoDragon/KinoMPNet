import julia
from julia.api import Julia
from utils import load_data
import numpy as np
from time import time

if __name__ == '__main__':
    jl = Julia(compiled_modules=False)
    j = julia.Julia(runtime="julia")

    model = 'acrobot'
    mu_u, sigma_u, mu_t, sigma_t = 0, 4, 0.02, 0.02
    # model = 'cartpole'
    # mu_u, sigma_u, mu_t, sigma_t = 0, 200, 0.02, 0.05
    cem = j.include("CEM_{}.jl".format(model))
    dx = j.include("{}.jl".format(model))

    data = load_data(model, 3)
    path = data['path']
    i_node = 0
    x = path[i_node]
    control, cost = [], []
    weights = np.array([1, 1, 0.3, 0.3])

    start = path[i_node]
    goal = path[i_node+1]

    cem(start, start, weights, mu_u, sigma_u, mu_t, sigma_t, False)
    for node in [i_node]:
        timer_start = time()
        for _ in range(100):
            u0, t0, mu_u, sigma_u, mu_t, sigma_t = \
                cem(start, goal, weights, mu_u, sigma_u, mu_t, sigma_t, False)
            x = dx(x, u0, int(np.floor(t0/0.002))+1, 0.002)
            for para in [mu_u, sigma_u, mu_t, sigma_t]:
                para[:-1] = para[1:]
            start = x.copy()
            loss = np.linalg.norm((np.array(start) - goal)*weights)
            cost.append((int(np.floor(t0/0.002))+1)*0.002)
            control.append(u0)
            print('loss:', loss)
            if(loss < 0.3):
                break
        timer_stop = time()
    print('cost:', cost, '\nsum', sum(cost), '\nref:', data['cost'][i_node])
    print('control:', control)
    print('time: ', timer_stop - timer_start)
