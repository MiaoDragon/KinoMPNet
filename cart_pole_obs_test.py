from sparse_rrt.planners import SST
from env.cartpole_obs import CartPoleObs
from sparse_rrt.systems import standard_cpp_systems
from sparse_rrt import _sst_module
import numpy as np
import time
# Create custom system
obs_list = [[-10., 10.],
            [10., 10.]]
obs_list = np.array(obs_list)
system = standard_cpp_systems.CartPoleObs(obs_list, 4.)
#system = CartPoleObs(obs_list)
# Create SST planner
min_time_steps = 2
max_time_steps = 4
integration_step = 0.002
max_iter = 2
goal_radius=1.5
random_seed=0
sst_delta_near=2.0
sst_delta_drain=1.2
planner = SST(
    state_bounds=system.get_state_bounds(),
    control_bounds=system.get_control_bounds(),
    distance=system.distance_computer(),
    start_state=np.array([0., 0., np.pi, 0.]),
    goal_state=np.array([5., 0., 0., 0.]),
    goal_radius=goal_radius,
    random_seed=0,
    sst_delta_near=sst_delta_near,
    sst_delta_drain=sst_delta_drain
)


low = []
high = []
state_bounds = system.get_state_bounds()
for i in range(len(state_bounds)):
    low.append(state_bounds[i][0])
    high.append(state_bounds[i][1])
end = np.array([5., 0., 0., 0.])
    
# Run planning and print out solution is some statistics every few iterations.
time0 = time.time()
for iteration in range(max_iter):
    #if iteration % 50 == 0:
    #    # from time to time use the goal
    #    sample = end
    #    planner.step_with_sample(system, sample, 20, 200, 0.002)
    #else:
    planner.step(system, min_time_steps, max_time_steps, integration_step)
    #    #sample = np.random.uniform(low=low, high=high)
    #print('iteration: %d' % (iteration))
    # interation: 0.002
    #planner.step_with_sample(system, sample, 2, 20, 0.01)
    
    if iteration % 100 == 0:
        solution = planner.get_solution()
        print("Solution: %s, Number of nodes: %s" % (planner.get_solution(), planner.get_number_of_nodes()))
        if solution is not None:
            break
print('time spent: %f' % (time.time() - time0))