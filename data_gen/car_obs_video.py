import sys
sys.path.append('../deps/sparse_rrt')
sys.path.append('..')
#from env.cartpole_obs import CartPoleObs
#from env.cartpole import CartPole
from sparse_rrt.systems import standard_cpp_systems
from sparse_rrt import _sst_module
import numpy as np
import time
from tools.pcd_generation import rectangle_pcd
import os
import pickle

MIN_X = -25
MAX_X = 25
MIN_Y = -35
MAX_Y = 35
obs_list = []
width = 8.
near = width * 2
car_width = 2.0
car_len = 1.0

def overlap(b1corner,b1axis,b1orign,b1dx,b1dy,b2corner,b2axis,b2orign,b2dx,b2dy):
    # this only checks overlap of b1 w.r.t. b2
    # a complete check should do in both directions
    b2ds = [b2dx, b2dy]
    for a in range(0,2):
        t=b1corner[0][0]*b2axis[a][0]+b1corner[0][1]*b2axis[a][1] # project corner to the axis by inner product

        tMin = t
        tMax = t
        for c in range(1,4):
            t = b1corner[c][0]*b2axis[a][0]+b1corner[c][1]*b2axis[a][1] # project corner to the axis by inner product
            # find range by [tMin, tMax]
            if t < tMin:
                tMin = t
            elif t > tMax:
                tMax = t
        # since b2 the other corners (corner 1, 2, 3) are larger than b2orign (project of corner 0 to axis)
        # specifically, the range is [b2orign[i], b2orign[i]+size(i)] (of the projected point by dot product)
        # we only need to compare tMax with b2orign[i], and tMin with size(i)+b2orign[i]
        if ((tMin > (b2ds[a] + b2orign[a])) or (tMax < b2orign[a])):
            return False

    return True


def enforce_bounds(state):
    '''

    check if state satisfies the bound
    apply threshold to velocity and angle
    return a new state toward which the bound has been enforced
    '''
    WIDTH = 2.0
    LENGTH = 1.0

    STATE_X = 0
    STATE_Y = 1

    STATE_THETA = 2 
    MIN_X = -25
    MAX_X = 25
    MIN_Y = -35
    MAX_Y = 35
    
    
    new_state = np.array(state)
    """
    if state[STATE_V] < MIN_V/30.:
        new_state[STATE_V] = MIN_V/30.
    elif state[STATE_V] > MAX_V/30.:
        new_state[STATE_V] = MAX_V/30.
    """
    if state[2] < -np.pi:
        new_state[2] += 2*np.pi
    elif state[2] > np.pi:
        new_state[2] -= 2*np.pi
    return new_state

def IsInCollision(x, obc, obc_width=4.):
    WIDTH = car_width # 1.0
    LENGTH = car_len # 2.0
    STATE_X = 0
    STATE_Y = 1
    STATE_THETA = 2
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
        # order: (left-bottom, right-bottom, right-upper, left-upper)
        # assume angle (state_theta) is clockwise
        robot_corner[0][j]=x[j]-X1[j]-Y1[j]
        robot_corner[1][j]=x[j]+X1[j]-Y1[j]
        robot_corner[2][j]=x[j]+X1[j]+Y1[j]
        robot_corner[3][j]=x[j]-X1[j]+Y1[j]

        # axis: horizontal and vertical
        robot_axis[0][j] = robot_corner[1][j] - robot_corner[0][j]
        robot_axis[1][j] = robot_corner[3][j] - robot_corner[0][j]

    #print('robot corners:')
    #print(robot_corner)
    length[0]=np.sqrt(robot_axis[0][0]*robot_axis[0][0]+robot_axis[0][1]*robot_axis[0][1])
    length[1]=np.sqrt(robot_axis[1][0]*robot_axis[1][0]+robot_axis[1][1]*robot_axis[1][1])
    # normalize the axis
    for i in range(0,2):
        for j in range(0,2):
            robot_axis[i][j]=robot_axis[i][j]/float(length[i])

    # obtain the projection of the left-bottom corner to the axis, to obtain the minimal projection length
    robot_orign[0]=robot_corner[0][0]*robot_axis[0][0]+ robot_corner[0][1]*robot_axis[0][1]
    robot_orign[1]=robot_corner[0][0]*robot_axis[1][0]+ robot_corner[0][1]*robot_axis[1][1]
    #print('robot orign:')
    #print(robot_orign)
    for i in range(len(obc)):
        cf=True

        obs_corner=np.zeros((4,2),dtype=np.float32)
        obs_axis=np.zeros((2,2),dtype=np.float32)
        obs_orign=np.zeros(2,dtype=np.float32)
        length2=np.zeros(2,dtype=np.float32)

        for j in range(0,2):
            # order: (left-bottom, right-bottom, right-upper, left-upper)
            obs_corner[0][j] = obc[i][j]
            obs_corner[1][j] = obc[i][2+j]
            obs_corner[2][j] = obc[i][2*2+j]
            obs_corner[3][j] = obc[i][3*2+j]
            
            # horizontal axis and vertical
            obs_axis[0][j] = obs_corner[1][j] - obs_corner[0][j]
            obs_axis[1][j] = obs_corner[3][j] - obs_corner[0][j]
            #obs_axis[0][j] = obs_corner[3][j] - obs_corner[0][j]
            #obs_axis[1][j] = obs_corner[1][j] - obs_corner[0][j]

        length2[0]=np.sqrt(obs_axis[0][0]*obs_axis[0][0]+obs_axis[0][1]*obs_axis[0][1])
        length2[1]=np.sqrt(obs_axis[1][0]*obs_axis[1][0]+obs_axis[1][1]*obs_axis[1][1])

        # normalize the axis
        for i1 in range(0,2):
            for j1 in range(0,2):
                obs_axis[i1][j1]=obs_axis[i1][j1]/float(length2[i1])

        # obtain the inner product of the left-bottom corner with the axis to obtain the minimal of projection value
        obs_orign[0]=obs_corner[0][0]*obs_axis[0][0]+ obs_corner[0][1]*obs_axis[0][1]  # dot product at 0-th corner
        obs_orign[1]=obs_corner[0][0]*obs_axis[1][0]+ obs_corner[0][1]*obs_axis[1][1]
        # do checking in both direction (b1 -> b2, b2 -> b1). If at least one shows not-overlapping, then it is not overlapping
        cf=overlap(robot_corner,robot_axis,robot_orign,car_width,car_len,obs_corner,obs_axis,obs_orign,width,width)
        cf=cf and overlap(obs_corner,obs_axis,obs_orign,width,width,robot_corner,robot_axis,robot_orign,car_width,car_len)
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



class CarVisualizer(Visualizer):
    def __init__(self, system, params):
        super(CarVisualizer, self).__init__(system, params)
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
                                           self.params['car_w']/2,0,\
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
                                                        -state[2]/np.pi * 180) + ax.transData
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
                                                        -state[2]/np.pi * 180) + ax.transData
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
        
        scat_state = ax.scatter([state[0]], [state[1]], [state[2]], c='blue')
        self.recs.append(scat_state)
        
        state = self.states[-1]
        
        ax.scatter([state[0]], [state[1]], [state[2]], c='red', marker='*')

        
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
                                                        -state[2]/np.pi * 180) + ax.transData
        self.recs[0].set_transform(t)

        # modify car_direction
        self.car_direction_patch.remove()
        self.car_direction = patches.Arrow(state[0],state[1],\
                                           self.params['car_w']/2,0,\
                                      linewidth=1.0, edgecolor='yellow')
        self.recs[1] = self.car_direction
        t = mpl.transforms.Affine2D().rotate_deg_around(state[0], state[1], \
                                                        -state[2]/np.pi * 180) + ax.transData
        self.recs[1].set_transform(t)
        self.car_direction_patch = ax.add_patch(self.car_direction)
        
        
        # print location of cart
        ax = self.ax2
        ax.set_xlim3d(-25, 25)
        ax.set_ylim3d(-35,35)
        ax.set_zlim3d(-np.pi,np.pi)

        self.recs[-1]._offsets3d = ([state[0]], [state[1]], [state[2]])

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
            #print('state: %d, remaining: %d' % (i, len(states)-i))
            action = actions[i]
            # number of steps for propagtion
            #num_steps = int(np.round(costs[i]/self.params['integration_step']))
            num_steps =100000
            while True:
                traj.append(np.array(s))
                #print("porpagating...")
                #print(s)
                #print('st:')
                #print(sT)
                s = self.system(s, action, self.params['integration_step'])
                if IsInCollision(s, obs_i, width):
                    print('state:')
                    print(s)
                    print('next state:')
                    print(states[i+1])
                
                assert not IsInCollision(s, obs_i, width)
                s = enforce_bounds(s)
                if np.linalg.norm(s - states[i+1]) <= 1e-8:
                    break

        traj = np.array(traj)
        print("animating...")
        # animate
        self.states = traj
        #print(traj)
        self.obs = obstacles
        print(len(self.states))
        self.total = len(self.states)
        ani = animation.FuncAnimation(self.fig, self._animate, range(0, len(self.states), 50),
                                      interval=self.dt, blit=True, init_func=self._init,
                                      repeat=True)
        return ani
    

    
# generate start and goal
print('here')
#obs_list = np.array(obs_list)
low = []
high = []
integration_step = 0.002    

# make sure start and goal are collision-free

writer=animation.FFMpegFileWriter(fps=int(1/integration_step))
for obs_idx in range(0,10):
    for p_idx_i in range(10):
        
        # Create custom system
        #obs_list = [[-10., -3.],
        #            [0., 3.],
        #            [10, -3.]]
        p_idx = np.random.randint(low=0, high=1000)
        #if os.path.exists('../car_env%d_path%d.mp4' % (obs_idx, p_idx)):
        #    continue
        file = open('../data/car_obs/obs_%d.pkl' % (obs_idx), 'rb')
        obs_list = pickle.load(file)
        file = open('../data/car_obs/obc_%d.pkl' % (obs_idx), 'rb')
        obc_list = pickle.load(file)
        print('generated.')
        print(obs_list.shape)
        # load path
        path = open('../data/car_obs/%d/path_%d.pkl' % (obs_idx, p_idx), 'rb')
        states = pickle.load(path)
        controls = open('../data/car_obs/%d/control_%d.pkl' % (obs_idx, p_idx), 'rb')
        actions = pickle.load(controls)
        costs = open('../data/car_obs/%d/cost_%d.pkl' % (obs_idx, p_idx), 'rb')
        costs = pickle.load(costs)


        # visualize
        params = {}
        params['car_w'] = car_width
        params['car_l'] = car_len
        params['obs_w'] = width
        params['obs_h'] = width
        params['integration_step'] = 0.002

        #system = _sst_module.RallyCar()
        propagate_system = _sst_module.Car()
        cpp_propagator = _sst_module.SystemPropagator()
        dynamics = lambda x, u, t: cpp_propagator.propagate(propagate_system, x, u, t)

        vis = CarVisualizer(dynamics, params)
        anim = vis.animate(np.array(states), np.array(actions), np.array(costs), obs_list)
        anim.save('../car_env%d_path%d.mp4' % (obs_idx, p_idx), writer=writer)
