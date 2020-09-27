import sys
sys.path.append('deps/sparse_rrt')
sys.path.append('.')

from sparse_rrt.planners import SST
#from env.cartpole_obs import CartPoleObs
#from env.cartpole import CartPole
from sparse_rrt.systems import standard_cpp_systems
from sparse_rrt import _sst_module
import numpy as np
import time
from tools.pcd_generation import rectangle_pcd


obs_list = []
width = 8.
near = width * 2
car_width = 1.0
car_len = 2.0
print('generating obs...')
for i in range(1):
    obs_single = []
    for j in range(5):
        low_x = -25 + width/2
        high_x = 25 - width/2
        low_y = -35 + width/2
        high_y = 35 - width/2
        while True:
            # randomly sample in the entire space
            obs = np.random.uniform(low=[low_x, low_y], high=[high_x, high_y])
            # check if it is near enough to previous obstacles
            too_near = False
            for k in range(len(obs_single)):
                if np.linalg.norm(obs - obs_single[k]) < near:
                    too_near = True
                    break
            if not too_near:
                break
            
        obs_single.append(obs)
    obs_single = np.array(obs_single)
    obs_list.append(obs_single)
obs_list = np.array(obs_list)
# convert from obs to point cloud
obc_list = rectangle_pcd(obs_list, width, 1400)
print('generated.')
print(obs_list.shape)

# Create custom system
#obs_list = [[-10., -3.],
#            [0., 3.],
#            [10, -3.]]
obs_list = obs_list[0]

obs_list_to_corner = []
for i in range(len(obs_list)):
    obs_list_to_corner_i = []
    obs_list_to_corner_i.append(obs_list[i][0]-width/2)
    obs_list_to_corner_i.append(obs_list[i][1]-width/2)
    obs_list_to_corner_i.append(obs_list[i][0]+width/2)
    obs_list_to_corner_i.append(obs_list[i][1]-width/2)
    obs_list_to_corner_i.append(obs_list[i][0]+width/2)
    obs_list_to_corner_i.append(obs_list[i][1]+width/2)
    obs_list_to_corner_i.append(obs_list[i][0]-width/2)
    obs_list_to_corner_i.append(obs_list[i][1]+width/2)
    obs_list_to_corner.append(obs_list_to_corner_i)
obs_list_to_corner = np.array(obs_list_to_corner)



def enforce_bounds(state):
    '''

    check if state satisfies the bound
    apply threshold to velocity and angle
    return a new state toward which the bound has been enforced
    '''
    M = 1450
    IZ = 2740
    LF = 1.3
    LR = 1.4
    R = 0.3
    IF = 1.8
    IR = 1.8
    H = 0.4
    B = 7
    C = 1.6
    D = 0.52
    
    WIDTH = 1.0
    LENGTH = 2.0
    
    CRBRAKE = 700
    CFACC = 1000

    STATE_X = 0
    STATE_Y = 1
    STATE_VX = 2
    STATE_VY = 3
    STATE_THETA = 4
    STATE_THETADOT = 5
    STATE_WF = 6
    STATE_WR = 7
    CONTROL_STA = 0
    CONTROL_TF = 1
    CONTROL_TR = 2

    MIN_X = -25
    AX_X = 25
    MIN_Y = -35
    MAX_Y = 35
    
    
    new_state = np.array(state)
    """
    if state[STATE_V] < MIN_V/30.:
        new_state[STATE_V] = MIN_V/30.
    elif state[STATE_V] > MAX_V/30.:
        new_state[STATE_V] = MAX_V/30.
    """
    if state[2] < -18:
        new_state[2] = -18
    elif state[2] > 18:
        new_state[2] = 18
    
    if state[3] < -18:
        new_state[3] = -18
    elif state[3] > 18:
        new_state[3] = 18
    
    if state[4] < -np.pi:
        new_state[4] += 2*np.pi
    elif state[4] > np.pi:
        new_state[4] -= 2*np.pi
    
    if state[5] < -17:
        new_state[5] = -17
    elif state[5] > 17:
        new_state[5] = 17
    
    if state[6] < -40:
        new_state[6] = -40
    elif state[6] > 40:
        new_state[6] = 40
        
    if state[7] < -40:
        new_state[7] = -40
    elif state[7] > 40:
        new_state[7] = 40
    return new_state

def overlap(b1corner,b1axis,b1orign,b2corner,b2axis,b2orign):
    for a in range(0,2):
        t=b1corner[0][0]*b2axis[a][0]+b1corner[0][1]*b2axis[a][1]

        tMin = t
        tMax = t
        for c in range(1,4):
            t = b1corner[c][0]*b2axis[a][0]+b1corner[c][1]*b2axis[a][1]
            if t < tMin:
                tMin = t
            elif t > tMax:
                tMax = t
        if ((tMin > (1+ b2orign[a])) or (tMax < b2orign[a])):
            return False

    return True

def IsInCollision(x, obc, obc_width=4.):
    M = 1450
    IZ = 2740
    LF = 1.3
    LR = 1.4
    R = 0.3
    IF = 1.8
    IR = 1.8
    H = 0.4
    B = 7
    C = 1.6
    D = 0.52
    
    WIDTH = car_width # 1.0
    LENGTH = car_len # 2.0
    
    CRBRAKE = 700
    CFACC = 1000

    STATE_X = 0
    STATE_Y = 1
    STATE_VX = 2
    STATE_VY = 3
    STATE_THETA = 4
    STATE_THETADOT = 5
    STATE_WF = 6
    STATE_WR = 7
    CONTROL_STA = 0
    CONTROL_TF = 1
    CONTROL_TR = 2

    MIN_X = -25
    MAX_X = 25
    MIN_Y = -35
    MAX_Y = 35
    if x[0] < MIN_X or x[0] > MAX_X or x[1] < MIN_Y or x[1] > MAX_Y:
        return True
        
    robot_corner=np.zeros((4,2),dtype=np.float32)
    robot_axis=np.zeros((2,2),dtype=np.float32)
    robot_orign=np.zeros(2,dtype=np.float32)
    length=np.zeros(2,dtype=np.float32)
    X1=np.zeros(2,dtype=np.float32)
    Y1=np.zeros(2,dtype=np.float32)

    X1[0]=np.cos(x[STATE_THETA])*(WIDTH/2.0)
    X1[1]=-np.sin(x[STATE_THETA])*(WIDTH/2.0)
    Y1[0]=np.sin(x[STATE_THETA])*(LENGTH/2.0)
    Y1[1]=np.cos(x[STATE_THETA])*(LENGTH/2.0)

    for j in range(0,2):
        robot_corner[0][j]=x[j]-X1[j]-Y1[j]
        robot_corner[1][j]=x[j]+X1[j]-Y1[j]
        robot_corner[2][j]=x[j]+X1[j]+Y1[j]
        robot_corner[3][j]=x[j]-X1[j]+Y1[j]

        robot_axis[0][j] = robot_corner[1][j] - robot_corner[0][j]
        robot_axis[1][j] = robot_corner[3][j] - robot_corner[0][j]

    length[0]=robot_axis[0][0]*robot_axis[0][0]+robot_axis[0][1]*robot_axis[0][1]
    length[1]=robot_axis[1][0]*robot_axis[1][0]+robot_axis[1][1]*robot_axis[1][1]
    #print "robot cornor"
    for i in range(0,2):
        for j in range(0,2):
            robot_axis[i][j]=robot_axis[i][j]/float(length[j])

    robot_orign[0]=robot_corner[0][0]*robot_axis[0][0]+ robot_corner[0][1]*robot_axis[0][1]
    robot_orign[1]=robot_corner[0][0]*robot_axis[1][0]+ robot_corner[0][1]*robot_axis[1][1]

    for i in range(len(obc)):
        cf=True

        obs_corner=np.zeros((4,2),dtype=np.float32)
        obs_axis=np.zeros((2,2),dtype=np.float32)
        obs_orign=np.zeros(2,dtype=np.float32)
        length2=np.zeros(2,dtype=np.float32)

        for j in range(0,2):
            #obs_corner[0][j]=obc[i][j]-X[j]-Y[j]
            #obs_corner[1][j]=obc[i][j]+X[j]-Y[j]
            #obs_corner[2][j]=obc[i][j]+X[j]+Y[j]
            #obs_corner[3][j]=obc[i][j]-X[j]+Y[j]
            obs_corner[0][j] = obc[i][j]
            obs_corner[1][j] = obc[i][2+j]
            obs_corner[2][j] = obc[i][2*2+j]
            obs_corner[3][j] = obc[i][3*2+j]
            

            obs_axis[0][j] = obs_corner[1][j] - obs_corner[0][j]
            obs_axis[1][j] = obs_corner[3][j] - obs_corner[0][j]

        length2[0]=obs_axis[0][0]*obs_axis[0][0]+obs_axis[0][1]*obs_axis[0][1]
        length2[1]=obs_axis[1][0]*obs_axis[1][0]+obs_axis[1][1]*obs_axis[1][1]

        for i1 in range(0,2):
            for j1 in range(0,2):
                obs_axis[i1][j1]=obs_axis[i1][j1]/float(length2[j1])


        obs_orign[0]=obs_corner[0][0]*obs_axis[0][0]+ obs_corner[0][1]*obs_axis[0][1]
        obs_orign[1]=obs_corner[0][0]*obs_axis[1][0]+ obs_corner[0][1]*obs_axis[1][1]

        cf=overlap(robot_corner,robot_axis,robot_orign,obs_corner,obs_axis,obs_orign)
        if cf==True:
            return True
    return False



# visualize the path
"""
Given a list of states, render the environment
"""
from numpy import sin, cos
import numpy as np
import matplotlib as mpl
mpl.use("Agg")

import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation

import matplotlib.patches as patches
from IPython.display import HTML
from visual.visualizer import Visualizer
from mpl_toolkits.mplot3d import Axes3D



class RallyCarVisualizer(Visualizer):
    def __init__(self, system, params):
        super(RallyCarVisualizer, self).__init__(system, params)
        self.dt = 2
        self.fig = plt.gcf()
        self.fig.set_figheight(5)
        self.fig.set_figwidth(10)
        self.ax1 = plt.subplot(121)
        #self.ax2 = plt.subplot(122)
        self.ax2 = self.fig.add_subplot(122, projection='3d')

        
    def _init(self):
        # clear the current ax
        ax = self.ax1
        ax.clear()
        ax.set_xlim(-25, 25)
        ax.set_ylim(-35, 35)
        # add patches
        state = self.states[0]
        self.car = patches.Rectangle((state[0]-self.params['car_w']/2,state[1]-self.params['car_l']/2),\
                                       self.params['car_w'],self.params['car_l'],\
                                      linewidth=.5,edgecolor='blue',facecolor='blue')
        self.car_direction = patches.Arrow(state[0],state[1],\
                                           0,self.params['car_l']/2,\
                                           linewidth=1.0, edgecolor='yellow')
        self.recs = []
        self.recs.append(self.car)
        self.recs.append(self.car_direction)
        for i in range(len(self.obs)):
            x, y = self.obs[i]
            obs = patches.Rectangle((x-self.params['obs_w']/2,y-params['obs_h']/2),\
                                       self.params['obs_w'],self.params['obs_h'],\
                                      linewidth=.5,edgecolor='black',facecolor='black')
            self.recs.append(obs)
            ax.add_patch(obs)
        # transform pole according to state
        t = mpl.transforms.Affine2D().rotate_deg_around(state[0], state[1], \
                                                        -state[4]/np.pi * 180) + ax.transData
        self.car.set_transform(t)
        ax.add_patch(self.car)
        self.car_direction.set_transform(t)
        self.car_direction_patch = ax.add_patch(self.car_direction)

        # add goal patch
        state = self.states[-1]
        self.car_goal = patches.Rectangle((state[0]-self.params['car_w']/2,state[1]-self.params['car_l']/2),\
                                       self.params['car_w'],self.params['car_l'],\
                                      linewidth=.5,edgecolor='red',facecolor='red')
        self.recs.append(self.car_goal)
        # transform pole according to state
        t = mpl.transforms.Affine2D().rotate_deg_around(state[0], state[1], \
                                                        -state[4]/np.pi * 180) + ax.transData
        self.car_goal.set_transform(t)
        ax.add_patch(self.car_goal)

        # state
        state = self.states[0]
        ax = self.ax2
        ax.clear()
        ax.set_xlim3d(-25, 25)
        ax.set_ylim3d(-35,35)
        ax.set_zlim3d(-np.pi,np.pi)

        
        dx = 1
        dtheta = 0.1
        feasible_points = []
        infeasible_points = []
        imin = 0
        imax = int(2*25./dx)
        jmin = 0
        jmax = int(2*35./dx)
        zmin = 0
        zmax = int(2*np.pi/dtheta)
        
        """
        for i in range(imin, imax):
            for j in range(jmin, jmax):
                for z in range(zmin, zmax):
                    x = np.array([dx*i-25, dx*j-35, 0., 0., dtheta*z-np.pi, 0., 0., 0.])
                    if IsInCollision(x, self.cc_obs):
                        infeasible_points.append(x)
                    else:
                        feasible_points.append(x)
        
        feasible_points = np.array(feasible_points)
        infeasible_points = np.array(infeasible_points)
        #ax.scatter(feasible_points[:,0], feasible_points[:,2], c='yellow')
        ax.scatter(infeasible_points[:,0], infeasible_points[:,1], infeasible_points[:,4], c='black')
        """
        
        scat_state = ax.scatter([state[0]], [state[1]], [state[4]], c='blue')
        self.recs.append(scat_state)
        
        state = self.states[-1]
        
        ax.scatter([state[0]], [state[1]], [state[4]], c='red', marker='*')

        
        # draw the goal region
        #ax = self.ax1
        # randomly sample several points
        
        return self.recs
    def _animate(self, i):
        ax = self.ax1
        ax.set_xlim(-25, 25)
        ax.set_ylim(-35, 35)
        state = self.states[i]
        self.recs[0].set_xy((state[0]-self.params['car_w']/2,state[1]-self.params['car_l']/2))
        t = mpl.transforms.Affine2D().rotate_deg_around(state[0], state[1], \
                                                        -state[4]/np.pi * 180) + ax.transData
        self.recs[0].set_transform(t)

        # modify car_direction
        self.car_direction_patch.remove()
        self.car_direction = patches.Arrow(state[0],state[1],\
                                           0,self.params['car_l']/2,\
                                      linewidth=1.0, edgecolor='yellow')
        self.recs[1] = self.car_direction
        t = mpl.transforms.Affine2D().rotate_deg_around(state[0], state[1], \
                                                        -state[4]/np.pi * 180) + ax.transData
        self.recs[1].set_transform(t)
        self.car_direction_patch = ax.add_patch(self.car_direction)
        
        
        # print location of cart
        ax = self.ax2
        ax.set_xlim3d(-25, 25)
        ax.set_ylim3d(-35,35)
        ax.set_zlim3d(-np.pi,np.pi)

        self.recs[-1]._offsets3d = ([state[0]], [state[1]], [state[4]])

        return self.recs


    def animate(self, states, actions, costs, obstacles):
        '''
        given a list of states, actions and obstacles, animate the robot
        '''

        new_obs_i = []
        obs_width = width
        for k in range(len(obstacles)):
            obs_pt = []
            obs_pt.append(obstacles[k][0]-obs_width/2)
            obs_pt.append(obstacles[k][1]-obs_width/2)
            obs_pt.append(obstacles[k][0]+obs_width/2)
            obs_pt.append(obstacles[k][1]-obs_width/2)
            obs_pt.append(obstacles[k][0]+obs_width/2)
            obs_pt.append(obstacles[k][1]+obs_width/2)
            obs_pt.append(obstacles[k][0]-obs_width/2)
            obs_pt.append(obstacles[k][1]+obs_width/2)
            new_obs_i.append(obs_pt)
        obs_i = new_obs_i
        self.cc_obs = obs_i

        # transform the waypoint states and actions into trajectory
        traj = []
        s = states[0]
        self.states = states
        self.obs = obstacles
        #self._init()
        #plt.show()
    
        for i in range(len(states)-1):
            print('state: %d, remaining: %d' % (i, len(states)-i))
            action = actions[i]
            # number of steps for propagtion
            #num_steps = int(np.round(costs[i]/self.params['integration_step']))
            num_steps =100000
            for j in range(num_steps):
                traj.append(np.array(s))
                #print("porpagating...")
                #print(s)
                #print('st:')
                #print(sT)
                s = self.system(s, action, self.params['integration_step'])
                assert not IsInCollision(s, obs_i, width)
                s = enforce_bounds(s)
                if np.linalg.norm(s - states[i+1]) <= 1e-3:
                    break

        traj = np.array(traj)
        print("animating...")
        # animate
        self.states = traj
        print(traj)
        self.obs = obstacles
        print(len(self.states))
        self.total = len(self.states)
        ani = animation.FuncAnimation(self.fig, self._animate, range(0, len(self.states), 10),
                                      interval=self.dt, blit=True, init_func=self._init,
                                      repeat=True)
        return ani
    

    
# generate start and goal
print('here')
#obs_list = np.array(obs_list)
system = standard_cpp_systems.RectangleObs(obs_list, width, 'rally_car')
#system = CartPoleObs(obs_list)
# Create SST planner
min_time_steps = 20
max_time_steps = 200
integration_step = 0.002
max_iter = 500000
goal_radius=8.
random_seed=0
sst_delta_near=1.2
sst_delta_drain=.5

low = []
high = []
state_bounds = system.get_state_bounds()
for i in range(len(state_bounds)):
    low.append(state_bounds[i][0])
    high.append(state_bounds[i][1])
    

# make sure start and goal are collision-free

path_id = 0
while True:
    # until a path is generated
    while True:
        start = np.random.uniform(low=low, high=high)
        end = np.random.uniform(low=low, high=high)
        print('start:')
        print(start)
        print('end:')
        print(end)
        # start and goal should be far enough if x-y
        if np.linalg.norm(start[:2] - end[:2]) <= 30:
            continue
        if not IsInCollision(start, obs_list_to_corner, width) and not IsInCollision(end, obs_list_to_corner, width):
            break
        #break
    start[[2,3,5,6,7]] = 0
    end[[2,3,5,6,7]] = 0
    start[6] = 0
    start[7] = 0
    end[6] = 0
    end[7] = 0
    print('start:')
    print(start)
    print('end:')
    print(end)    

    # planning


    #start[2] = 0.
    #start[3] = 0.

    #end[1] = 0.
    #end[3] = 0.
    planner = SST(
        state_bounds=system.get_state_bounds(),
        control_bounds=system.get_control_bounds(),
        distance=system.distance_computer(),
        start_state=start,
        goal_state=end,
        goal_radius=goal_radius,
        random_seed=0,
        sst_delta_near=sst_delta_near,
        sst_delta_drain=sst_delta_drain
    )

    # Run planning and print out solution is some statistics every few iterations.

    time0 = time.time()


    for iteration in range(max_iter):
        #if iteration % 50 == 0:
        #    # from time to time use the goal
        #    sample = end
        #    planner.step_with_sample(system, sample, 20, 200, 0.002)
        #else:
        if iteration % int(0.1 * max_iter) == 0:        
            print('iteration: %d/%d' % (iteration, max_iter))
        steer_start, steer_end = planner.step_with_output(system, min_time_steps, max_time_steps, integration_step)
        #if planner.get_solution():
        #    break
        #print('start:')
        #print(steer_start)
        #print('end:')
        #print(steer_end)
        #ax.scatter(steer_start[0], steer_start[1], steer_start[4], c='red')
        #ax.scatter(steer_end[0], steer_end[1], steer_end[4], c='blue')

        #    #sample = np.random.uniform(low=low, high=high)
        #print('iteration: %d' % (iteration))
        # interation: 0.002
        #planner.step_with_sample(system, sample, 2, 20, 0.01)

        #if iteration % 100 == 0:
    solution = planner.get_solution()
    print("Solution: %s, Number of nodes: %s" % (planner.get_solution(), planner.get_number_of_nodes()))

    print('time spent: %f' % (time.time() - time0))

    if solution is None:
        # generate the next start-goal pair
        continue


    # visualize
    params = {}
    params['car_w'] = car_width
    params['car_l'] = car_len
    params['obs_w'] = width
    params['obs_h'] = width
    params['integration_step'] = integration_step

    #system = _sst_module.RallyCar()
    cpp_propagator = _sst_module.SystemPropagator()
    dynamics = lambda x, u, t: cpp_propagator.propagate(system, x, u, t)

    vis = RallyCarVisualizer(dynamics, params)
    states, actions, costs = solution
    anim = vis.animate(np.array(states), np.array(actions), np.array(costs), obs_list)
    writer=animation.FFMpegFileWriter(fps=int(1/integration_step))
    anim.save('rally_car_test_path_%d.mp4' % (path_id), writer=writer)
    path_id += 1
    if path_id == 5:
        break
