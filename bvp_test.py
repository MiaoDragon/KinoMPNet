from ctypes import *
#import faulthandler
#faulthandler.enable()
#ctypes.cdll.LoadLibrary('')
lib1 = CDLL("/home/yinglong/Documents/kinodynamic/sparse_rrt/deps/trajopt/build/lib/libsco.so")
lib2 = CDLL("/home/yinglong/Documents/kinodynamic/sparse_rrt/deps/trajopt/build/lib/libutils.so")
#from env.cartpole import CartPole
import sys
sys.path.append('deps/sparse_rrt')
import sparse_rrt
from sparse_rrt.systems import standard_cpp_systems
from sparse_rrt import _sst_module
import numpy as np
import time
import matplotlib.pyplot as plt
from sparse_rrt.systems.pendulum import Pendulum
#obs_list = np.array(obs_list)
#system = standard_cpp_systems.PSOPTCartPole()
_system = sparse_rrt._sst_module.PSOPTPendulum()
bvp_solver = _sst_module.PSOPTBVPWrapper(_system, 2, 1, 0)
#start = np.array([0., 0.])
#end = np.array([np.pi/2, 0.])
low = []
high = []
state_bounds = _system.get_state_bounds()
for i in range(len(state_bounds)):
    low.append(state_bounds[i][0])
    high.append(state_bounds[i][1])

for N in range(10):
    start = np.random.uniform(low=low, high=high)
    end = np.random.uniform(low=low, high=high)
    start[1] = 0.
    end[1] = 0.

    state, control, times = bvp_solver.solve(start, end, 100, 20, 200, 0.002)

    #solution = bvp_solver.solve(start, goal)
    #print(solution)
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_autoscale_on(True)
    hl, = ax.plot([], [], 'b')
    hl_real, = ax.plot([], [], 'r')

    def update_line(h, ax, new_data):
        h.set_xdata(np.append(h.get_xdata(), new_data[0]))
        h.set_ydata(np.append(h.get_ydata(), new_data[1]))
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()

    for i in range(len(state)):
        update_line(hl, ax, state[i])
    system = Pendulum()

    update_line(hl_real, ax, state[0])
    start = state[0]
    integration_step = 0.002
    print('states:')
    print(state)
    print('control:')
    print(control)
    print('times:')
    print(times)
    integration_step = 0.0001
    real_state = []
    for i in range(len(times)):
        num_steps = int(np.floor(times[i] / integration_step))
        if num_steps == 0:
            start = system.propagate(start, control[i], 1, times[i])
        else:
            start = system.propagate(start, control[i], num_steps, integration_step)
            start = system.propagate(start, control[i], 1, times[i] - num_steps * integration_step)
        real_state.append(start)

        update_line(hl_real, ax, start)
    print('real state:')
    print(real_state)
