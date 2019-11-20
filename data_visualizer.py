from ctypes import *
import sys
sys.path.append('deps/sparse_rrt')

#import faulthandler
#faulthandler.enable()
#ctypes.cdll.LoadLibrary('')
lib1 = CDLL("/home/yinglong/Documents/kinodynamic/sparse_rrt/deps/trajopt/build/lib/libsco.so")
lib2 = CDLL("/home/yinglong/Documents/kinodynamic/sparse_rrt/deps/trajopt/build/lib/libutils.so")
#from env.cartpole import CartPole
import sparse_rrt
from sparse_rrt.systems import standard_cpp_systems
from sparse_rrt import _sst_module
import numpy as np
import time
import matplotlib.pyplot as plt
from sparse_rrt.systems.pendulum import Pendulum
import pickle
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

for i in range(10):
    f = open('data/pendulum/0/path_%d.pkl' % (i), 'rb')
    state = pickle.load(f)

    f = open('data/pendulum/0/control_%d.pkl' % (i), 'rb')
    control = pickle.load(f)

    f = open('data/pendulum/0/cost_%d.pkl' % (i), 'rb')
    times = pickle.load(f)


    #state, control, times = bvp_solver.solve(start, end, 100, 20, 200, 0.002)

    #solution = bvp_solver.solve(start, goal)
    #print(solution)
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_autoscale_on(True)
    hl, = ax.plot([], [], 'b')
    hl_real, = ax.plot([], [], 'r')
    hl_bvp, = ax.plot([], [], 'g')
    def update_line(h, ax, new_data):
        h.set_xdata(np.append(h.get_xdata(), new_data[0]))
        h.set_ydata(np.append(h.get_ydata(), new_data[1]))
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()

    #for i in range(len(state)):
    #    update_line(hl, ax, state[i])
    system = Pendulum()


    integration_step = 0.002

    update_line(hl, ax, state[0])
    state_data = [state[0]]
    start = state[0]
    for i in range(len(times)):
        num_steps = int(np.floor(times[i] / integration_step))
        if num_steps == 0:
            start = system.propagate(start, control[i], 1, times[i])
        else:
            start = system.propagate(start, control[i], num_steps, integration_step)
            start = system.propagate(start, control[i], 1, times[i] - num_steps * integration_step)
        state_data.append(start)

        update_line(hl, ax, start)

    start = state[0]

    print('states:')
    print(state)
    print('control:')
    print(control)
    print('times:')
    print(times)
    plt.waitforbuttonpress()
    #for i in range(len(times)):
    #    num_steps = int(times[i] / integration_step)
    #    start = system.propagate(start, control[i], num_steps, integration_step)
    #    update_line(hl_real, ax, start)
    #### try to use each waypoint from the solution to guide the search first
    bvp_traj_state = state[0]
    real_traj_state = state[0]
    for i in range(len(times)):
        ##### BVP solver solves Traj Opt first to obtain candidate controls
        bvp_states, bvp_controls, bvp_times = bvp_solver.solve(real_traj_state, state[i+1], 500, 20, 100, 0.002)
        for j in range(len(bvp_states)):
            update_line(hl_bvp, ax, bvp_states[j])
        print('after bvp solver:')
        print('states:')
        print(bvp_states)
        print('controls:')
        print(bvp_controls)
        print('times:')
        print(bvp_times)
        plt.waitforbuttonpress()
        real_state = []

        ####### System tries to simulate the given trajectory
        start = real_traj_state
        update_line(hl_real, ax, start)
        for j in range(len(bvp_times)):
            num_steps = int(np.floor(bvp_times[j] / integration_step))
            if num_steps == 0:
                start = system.propagate(start, bvp_controls[j], 1, bvp_times[j])
            else:
                start = system.propagate(start, bvp_controls[j], num_steps, integration_step)
                start = system.propagate(start, bvp_controls[j], 1, bvp_times[j] - num_steps * integration_step)
            real_state.append(start)

            update_line(hl_real, ax, start)
        # update last real state, this also serves as the new BVP start state
        real_traj_state = start
        print('after real state:')
        print(real_state)
        plt.waitforbuttonpress()
