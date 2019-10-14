from sparse_rrt import _sst_module
from sparse_rrt.systems import standard_cpp_systems
import numpy as np
import time

from sparse_rrt.systems.acrobot import Acrobot, AcrobotDistance
from sparse_rrt.systems.point import Point

# this is a test code for SST in CartPole environment
system = standard_cpp_systems.Point()

planner = _sst_module.SSTWrapper(
    state_bounds=system.get_state_bounds(),
    control_bounds=system.get_control_bounds(),
    distance=system.distance_computer(),
    start_state=np.array([0., 0.]),
    goal_state=np.array([9., 9.]),
    goal_radius=0.5,
    random_seed=0,
    sst_delta_near=0.4,
    sst_delta_drain=0.2
)

number_of_iterations = 10000

min_time_steps = 20
max_time_steps = 200
integration_step = 0.002

print("Starting the planner.")

start_time = time.time()
state_bounds = system.get_state_bounds()
low = []
high = []
for i in range(len(state_bounds)):
    low.append(state_bounds[i][0])
    high.append(state_bounds[i][1])

for iteration in range(number_of_iterations):
    # randomly sample points
    sample = np.random.uniform(low=low, high=high)
    new_sample = planner.step_with_sample(system, sample, min_time_steps, max_time_steps, integration_step)
    if iteration % 100000 == 0:
        solution = planner.get_solution()

        if solution is None:
            solution_cost = None
        else:
            solution_cost = np.sum(solution[2])

        print("Time: %.2fs, Iterations: %d, Nodes: %d, Solution Quality: %s" %
              (time.time() - start_time, iteration, planner.get_number_of_nodes(), solution_cost))

path, controls, costs = planner.get_solution()
solution_cost = np.sum(costs)

print("Time: %.2fs, Iterations: %d, Nodes: %d, Solution Quality: %f" %
      (time.time() - start_time, number_of_iterations, planner.get_number_of_nodes(), solution_cost))


# printout the path and controls
print('solution path:')
print(path)
print('solution control:')
print(controls)
print('solution costs:')
print(costs)



"""
solution path:
[[ 0.          0.        ]
 [-0.6370412   2.87494444]
 [-0.17913047  4.0765825 ]
 [-0.65570388  4.91189047]
 [-0.22485279  6.43088118]
 [ 1.16924928  9.99335223]
 [ 2.28869902  9.76708703]
 [ 3.0422293   9.58632389]
 [ 4.15041416  9.8207588 ]
 [ 6.05159075  9.16934582]
 [ 8.18297696  9.25270438]
 [ 8.52685443  9.13393694]]
solution control:
[[ 9.88146895  1.78885684]
 [ 6.91360276  1.20671283]
 [ 8.29049366  2.08926944]
 [ 9.39829007  1.29441267]
 [ 9.96233161  1.19778629]
 [ 9.84558116 -0.19943492]
 [ 6.79744304 -0.23543941]
 [ 9.93605831  0.20847488]
 [ 9.75572526 -0.33010014]
 [ 9.35533184  0.0390901 ]
 [ 8.66213307 -0.3325506 ]]
solution costs:
[0.298 0.186 0.116 0.168 0.384 0.116 0.114 0.114 0.206 0.228 0.042]
"""



system = standard_cpp_systems.CartPole()
state_bounds = system.get_state_bounds()
low = []
high = []
for i in range(len(state_bounds)):
    low.append(state_bounds[i][0])
    high.append(state_bounds[i][1])
start = np.random.uniform(low=low, high=high)
end = np.random.uniform(low=low, high=high)

planner = _sst_module.SSTWrapper(
    state_bounds=system.get_state_bounds(),
    control_bounds=system.get_control_bounds(),
    distance=system.distance_computer(),
    start_state=start,
    goal_state=end,
    goal_radius=1.5,
    random_seed=0,
    sst_delta_near=2.,
    sst_delta_drain=1.2
)
min_time_steps = 10
max_time_steps = 50
integration_step = 0.02
number_of_iterations = 400000
for iteration in range(number_of_iterations):
    if iteration % 100 == 0:
        # from time to time use the goal
        sample = end
    else:
        sample = np.random.uniform(low=low, high=high)
    planner.step_with_sample(system, sample, min_time_steps, max_time_steps, integration_step)
path, controls, costs = planner.get_solution()
solution_cost = np.sum(costs)

print("Time: %.2fs, Iterations: %d, Nodes: %d, Solution Quality: %f" %
      (time.time() - start_time, number_of_iterations, planner.get_number_of_nodes(), solution_cost))


# printout the path and controls
print('solution path:')
print(path)
print('solution control:')
print(controls)
print('solution costs:')
print(costs)

"""
solution path:
[[-20.           0.           3.14         0.        ]
 [-17.33678848  11.80413067  -2.48903442   2.        ]
 [-13.33030012  16.98129937  -1.92903442   2.        ]
 [  1.87701961  27.00197333  -0.54586192   1.70843411]
 [ 11.50649553  16.61234684   0.32560687   2.        ]
 [ 19.67573781   3.90264169   1.76979015   0.98770021]
 [ 20.21600244  -0.12029982   1.90038477  -0.09413916]]
solution control:
[[ 298.46432414]
 [ 271.96680053]
 [ 252.98090694]
 [-261.53276242]
 [-260.30192018]
 [-221.73883191]]
solution costs:
[0.46 0.28 0.7  0.44 0.8  0.26]
"""
