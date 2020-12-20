## from ctypes import *
#ctypes.cdll.LoadLibrary('')
#lib1 = CDLL("deps/sparse_rrt/deps/trajopt/build/lib/libsco.so")
#lib2 = CDLL("deps/sparse_rrt/deps/trajopt/build/lib/libutils.so")

import sys
sys.path.append('deps/sparse_rrt')
sys.path.append('.')
import matplotlib as mpl
mpl.use('Agg')

from sparse_rrt.planners import SST
#from env.cartpole_obs import CartPoleObs
#from env.cartpole import CartPole
#from sparse_rrt.systems.cartpole import Cartpole
from sparse_rrt.systems import standard_cpp_systems
from sparse_rrt import _sst_module
import numpy as np
import time
from tools.pcd_generation import rectangle_pcd
from plan_utility.line_line_cc import *
import pickle
obs_list = []
LENGTH = 20.
width = 6.
near = width * 1.2
# convert from obs to point cloud
# load generated point cloud
obs_list_total = []
obc_list_total = []
for i in range(10):
    file = open('/media/arclabdl1/HD1/YLmiao/YLmiao_from_unicorn/YLmiao/data/kinodynamic/cartpole_obs/obs_%d.pkl' % (i), 'rb')
    obs_list_total.append(pickle.load(file))
    file = open('/media/arclabdl1/HD1/YLmiao/YLmiao_from_unicorn/YLmiao/data/kinodynamic/cartpole_obs/obc_%d.pkl' % (i), 'rb')
    obc_list_total.append(pickle.load(file))

#[(0, 932), (1, 935), (2, 923), (8, 141), (5,931), (7, 927)]
# (5,931), (6, 286)

import sys
obs_idx = int(sys.argv[1])

p_idx = int(sys.argv[2])


print('obs_idx: ')
print(obs_idx)
print('p_idx:')
print(p_idx)
        

# Create custom system
#obs_list = [[-10., -3.],
#            [0., 3.],
#            [10, -3.]]
obs_list = obs_list_total[obs_idx]
obc_list = obc_list_total[obs_idx]
print('generated.')
print(obs_list.shape)
# load path
path = open('/media/arclabdl1/HD1/YLmiao/YLmiao_from_unicorn/YLmiao/data/kinodynamic/cartpole_obs/%d/path_%d.pkl' % (obs_idx, p_idx), 'rb')
path = pickle.load(path)
controls = open('/media/arclabdl1/HD1/YLmiao/YLmiao_from_unicorn/YLmiao/data/kinodynamic/cartpole_obs/%d/control_%d.pkl' % (obs_idx, p_idx), 'rb')
controls = pickle.load(controls)
costs = open('/media/arclabdl1/HD1/YLmiao/YLmiao_from_unicorn/YLmiao/data/kinodynamic/cartpole_obs/%d/cost_%d.pkl' % (obs_idx, p_idx), 'rb')
costs = pickle.load(costs)
sgs = open('/media/arclabdl1/HD1/YLmiao/YLmiao_from_unicorn/YLmiao/data/kinodynamic/cartpole_obs/%d/start_goal_%d.pkl' % (obs_idx, p_idx), 'rb')
sgs = pickle.load(sgs)

print(sgs)



from plan_utility.line_line_cc import line_line_cc

def IsInCollision(x, obc, obc_width=4.):
    I = 10
    L = 2.5
    M = 10
    m = 5
    g = 9.8
    H = 0.5

    STATE_X = 0
    STATE_V = 1
    STATE_THETA = 2
    STATE_W = 3
    CONTROL_A = 0

    MIN_X = -30
    MAX_X = 30
    MIN_V = -40
    MAX_V = 40
    MIN_W = -2
    MAX_W = 2


    if x[0] < MIN_X or x[0] > MAX_X:
        return True

    H = 0.5
    pole_x1 = x[0]
    pole_y1 = H
    pole_x2 = x[0] + L * np.sin(x[2])
    pole_y2 = H + L * np.cos(x[2])


    for i in range(len(obc)):
        for j in range(0, 8, 2):
            x1 = obc[i][j]
            y1 = obc[i][j+1]
            x2 = obc[i][(j+2) % 8]
            y2 = obc[i][(j+3) % 8]
            if line_line_cc(pole_x1, pole_y1, pole_x2, pole_y2, x1, y1, x2, y2):
                return True
    return False

def enforce_bounds(state):
    '''

    check if state satisfies the bound
    apply threshold to velocity and angle
    return a new state toward which the bound has been enforced
    '''
    I = 10
    L = 2.5
    M = 10
    m = 5
    g = 9.8
    H = 0.5  # cart
    # define the name for each state index and action index
    STATE_X, STATE_V, STATE_THETA, STATE_W = 0, 1, 2, 3
    CONTROL_A = 0
    # define boundary
    MIN_X = -30
    MAX_X = 30
    MIN_V = -40
    MAX_V = 40
    MIN_W = -2
    MAX_W = 2
    # obstacle information
    OBS_W = 4
    new_state = np.array(state)
    """
    if state[STATE_V] < MIN_V/30.:
        new_state[STATE_V] = MIN_V/30.
    elif state[STATE_V] > MAX_V/30.:
        new_state[STATE_V] = MAX_V/30.
    """

    if state[STATE_V] < MIN_V:
        new_state[STATE_V] = MIN_V
    elif state[STATE_V] > MAX_V:
        new_state[STATE_V] = MAX_V

    if state[STATE_THETA] < -np.pi:
        new_state[STATE_THETA] += 2*np.pi
    elif state[STATE_THETA] > np.pi:
        new_state[STATE_THETA] -= 2*np.pi

    if state[STATE_W] < MIN_W:
        new_state[STATE_W] = MIN_W
    elif state[STATE_W] > MAX_W:
        new_state[STATE_W] = MAX_W
    return new_state

def wrap_angle(x, system):
    circular = system.is_circular_topology()
    res = np.array(x)
    for i in range(len(x)):
        if circular[i]:
            # use our previously saved version
            res[i] = x[i] - np.floor(x[i] / (2*np.pi))*(2*np.pi)
            if res[i] > np.pi:
                res[i] = res[i] - 2*np.pi
    return res


from visual.visualizer import Visualizer
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
class CartpoleVisualizer(Visualizer):
    def __init__(self, system, params):
        super(CartpoleVisualizer, self).__init__(system, params)
        self.dt = 2
        #self.fig1 = plt.figure(figsize=(25,6))
        #self.fig2 = plt.figure(figsize=(25,20))
        #self.ax1 = self.fig1.add_subplot(1,1,1)
        #self.ax2 = self.fig2.add_subplot(1,1,1)
        self.fig = plt.figure(figsize=(30,26))
        (self.ax1, self.ax2) = self.fig.subplots(2, 1, gridspec_kw={'height_ratios': [6, 20]})
    def _init(self):
        ##### handle the animation
        # clear the current ax
        print("in init")
        ax = self.ax1
        ax.clear()
        # add goal
        goal_state = self.sgs[1]
        goal_pole = patches.Rectangle((goal_state[0]-self.params['pole_w']/2,self.params['cart_h']),\
                                       self.params['pole_w'],self.params['pole_l'],\
                                      linewidth=1.,edgecolor=self.color_dict['pole_goal_color'],\
                                      facecolor=self.color_dict['pole_goal_color'])
        goal_cart = patches.Rectangle((goal_state[0]-self.params['cart_w']/2,0),\
                                       self.params['cart_w'],self.params['cart_h'],\
                                      linewidth=1.,edgecolor=self.color_dict['pole_goal_color'],\
                                      facecolor=self.color_dict['pole_goal_color'])
        # transform pole according to state
        t = mpl.transforms.Affine2D().rotate_deg_around(goal_state[0], self.params['cart_h'], \
                                                        -goal_state[2]/np.pi * 180) + ax.transData
        goal_pole.set_transform(t)
        ax.add_patch(goal_pole)
        ax.add_patch(goal_cart)
        
        
        # add patches
        state = self.states[0]
        self.pole = patches.Rectangle((state[0]-self.params['pole_w']/2,self.params['cart_h']),\
                                       self.params['pole_w'],self.params['pole_l'],\
                                      linewidth=1.,edgecolor=self.color_dict['pole_intermediate_color'],\
                                      facecolor=self.color_dict['pole_intermediate_color'])
        self.cart = patches.Rectangle((state[0]-self.params['cart_w']/2,0),\
                                       self.params['cart_w'],self.params['cart_h'],\
                                      linewidth=1.,edgecolor=self.color_dict['cart_intermediate_color'],\
                                      facecolor=self.color_dict['cart_intermediate_color'])
        self.recs = []
        self.recs.append(self.pole)
        self.recs.append(self.cart)
        for i in range(len(self.obs)):
            x, y = self.obs[i]
            obs = patches.Rectangle((x-self.params['obs_w']/2,y-params['obs_h']/2),\
                                       self.params['obs_w'],self.params['obs_h'],\
                                      linewidth=.5,edgecolor=self.color_dict['obstacle_color'],\
                                      facecolor=self.color_dict['obstacle_color'])
            self.recs.append(obs)
            ax.add_patch(obs)
        # transform pole according to state
        t = mpl.transforms.Affine2D().rotate_deg_around(state[0], self.params['cart_h'], \
                                                        -state[2]/np.pi * 180) + ax.transData
        self.pole.set_transform(t)
        ax.add_patch(self.pole)
        ax.add_patch(self.cart)

        #### handle search space
        ax = self.ax2
        ax.clear()
        ax.set_xlim(-30, 30)
        ax.set_ylim(-np.pi, np.pi)

        dx = 1
        dtheta = 0.1
        feasible_points = []
        infeasible_points = []
        imin = 0
        imax = int(2*30./dx)
        jmin = 0
        jmax = int(2*np.pi/dtheta)

        for i in range(imin, imax):
            for j in range(jmin, jmax):
                x = np.array([dx*i-30, 0., dtheta*j-np.pi, 0.])
                if IsInCollision(x, self.cc_obs):
                    infeasible_points.append(x)
                else:
                    feasible_points.append(x)
        feasible_points = np.array(feasible_points)
        infeasible_points = np.array(infeasible_points)
        
        print('feasible points')
        print(feasible_points)
        print('infeasible points')
        print(infeasible_points)
        #scat_feas =ax.scatter(feasible_points[:,0], feasible_points[:,2], c='yellow')
        scat_infeas = ax.scatter(infeasible_points[:,0], infeasible_points[:,2], c=self.color_dict['obstacle_color'])

        #self.recs.append(scat_feas)
        self.recs.append(scat_infeas)
        goal_scat_state = ax.scatter(goal_state[0], goal_state[2], c=self.color_dict['state_goal_color'])

        scat_state = ax.scatter(state[0], state[2], c=self.color_dict['state_intermediate_color'])
        self.recs.append(scat_state)
        print("after init")

        return self.recs
    
    def _animate(self, i):
        print('animating, frame %d/%d' % (i, self.total))
        
        ax = self.ax1
        ax.set_xlim(-30, 30)
        ax.set_ylim(-6, 6)
        ax.axis('off')
        state = self.states[i]
        self.recs[0].set_xy((state[0]-self.params['pole_w']/2,self.params['cart_h']))
        t = mpl.transforms.Affine2D().rotate_deg_around(state[0], self.params['cart_h'], \
                                                        -state[2]/np.pi * 180) + ax.transData
        self.recs[0].set_transform(t)
        self.recs[1].set_xy((state[0]-self.params['cart_w']/2,0))


        # handle search space
        ax = self.ax2
        ax.axis('off')

        ax.set_xlim(-30, 30)
        ax.set_ylim(-np.pi, np.pi)
        self.recs[-1].set_offsets([state[0], state[2]])
        # print location of cart
        return self.recs


    def animate(self, states, actions, costs, obstacles, sg, wrap_system):
        '''
        given a list of states, actions and obstacles, animate the robot
        '''

        new_obs_i = []
        obs_width = 4.0
        for k in range(len(obstacles)):
            obs_pt = []
            obs_pt.append(obstacles[k][0]-obs_width/2)
            obs_pt.append(obstacles[k][1]-obs_width/2)
            obs_pt.append(obstacles[k][0]-obs_width/2)
            obs_pt.append(obstacles[k][1]+obs_width/2)
            obs_pt.append(obstacles[k][0]+obs_width/2)
            obs_pt.append(obstacles[k][1]+obs_width/2)
            obs_pt.append(obstacles[k][0]+obs_width/2)
            obs_pt.append(obstacles[k][1]-obs_width/2)
            new_obs_i.append(obs_pt)
        obs_i = new_obs_i
        self.cc_obs = obs_i

        # transform the waypoint states and actions into trajectory
        traj = []
        s = states[0]
        for i in range(len(states)-1):
            print('state: %d, remaining: %d' % (i, len(states)-i))
            
            
            action = actions[i]
            # number of steps for propagtion
            #num_steps = int(np.round(costs[i]/self.params['integration_step']))
            num_steps = 100000
            for j in range(num_steps):
                traj.append(np.array(s))
                #print("porpagating...")
                #print(s)
                #print('st:')
                #print(sT)
                s = self.system(s, action, self.params['integration_step'])
                assert not IsInCollision(s, obs_i)
                if np.linalg.norm(s - states[i+1]) <= 1e-3:
                    break

        return np.array(traj)
    
    
    # plot the trajectory
    def plot(self, traj, obstacles, sg, color_dict, wrap_system):
        self.fig1 = plt.figure(figsize=(25,6))
        self.fig2 = plt.figure(figsize=(25,20))
        self.ax1 = self.fig1.add_subplot(1,1,1)
        self.ax2 = self.fig2.add_subplot(1,1,1)
        self.color_dict = color_dict
        print("animating...")
        # animate
        self.states = traj
        self.obs = obstacles
        print(len(self.states))
        self.total = len(self.states)
        self._init()
        
        to_plot_list_x = []
        to_plot_list_y = []
        traj_recs = []
        plot_step_sz = 50
        for i in list(range(1,len(traj))) + [0]:
            if i % plot_step_sz == 0:
                # plot the scene change
                if i == 0:
                    pole_color = color_dict["pole_start_color"]
                    cart_color = color_dict["cart_start_color"]
                elif i+plot_step_sz >= len(traj):
                    pole_color = color_dict["pole_goal_color"]
                    cart_color = color_dict["cart_goal_color"]
                else:
                    pole_color = color_dict["pole_intermediate_color"]
                    cart_color = color_dict["cart_intermediate_color"]
                state = traj[i]
                pole = patches.Rectangle((state[0]-self.params['pole_w']/2,self.params['cart_h']),\
                                               self.params['pole_w'],self.params['pole_l'],\
                                              linewidth=.5,edgecolor=pole_color,facecolor=pole_color)
                cart = patches.Rectangle((state[0]-self.params['cart_w']/2,0),\
                                               self.params['cart_w'],self.params['cart_h'],\
                                              linewidth=.5,edgecolor=cart_color,facecolor=cart_color)

                traj_recs.append(pole)
                traj_recs.append(cart)
                ax = self.ax1
                ax.set_xlim(-25, 25)
                ax.set_ylim(-6, 6)
                ax.add_patch(pole)
                ax.add_patch(cart)
                pole.set_xy((state[0]-self.params['pole_w']/2,self.params['cart_h']))
                t = mpl.transforms.Affine2D().rotate_deg_around(state[0], self.params['cart_h'], \
                                                                -state[2]/np.pi * 180) + ax.transData
                pole.set_transform(t)
                cart.set_xy((state[0]-self.params['cart_w']/2,0))
                ax.axis('off')

            # plot in the state space    
            ax = self.ax2
            traj_to_plot = wrap_angle(traj[i], wrap_system)
            scat_state = ax.scatter(traj_to_plot[0], traj_to_plot[2], c=color_dict['state_intermediate_color'], s=25.0)
        colors = [color_dict["state_start_color"], color_dict["state_goal_color"]]
                    
        # start and goal
        ax = self.ax2
        scat_state = ax.scatter(sg[0][0], sg[0][2], c=color_dict["state_start_color"], s=200.0)
        scat_state = ax.scatter(sg[1][0], sg[1][2], c=color_dict["state_goal_color"], s=200.0, marker='*')
        plt.subplots_adjust(wspace=0, hspace=0)

        ax.axis('off')

        #plt.savefig("cartpole_mpnettree_%d_p_%d.png" % (obs_idx, p_idx), bbox_inches='tight')
        return self.fig1, self.fig2
    
    def gen_video(self, traj, obs_list, sgs, color_dict, system):
        self.states = traj
        self.obs_list = obs_list
        self.obs = obs_list
        self.color_dict = color_dict
        self.total = len(self.states)
        self.sgs = sgs

        ani = animation.FuncAnimation(self.fig, self._animate, range(0, len(self.states)),
                                      interval=self.dt, blit=True, init_func=self._init,
                                      repeat=True)
        return ani
        
    
params = {}
params['pole_l'] = 2.5
params['pole_w'] = 0.01
params['cart_w'] = 1.
params['cart_h'] = 0.5
params['obs_w'] = 4
params['obs_h'] = 4
params['integration_step'] = 0.002
system = _sst_module.PSOPTCartPole()
cpp_propagator = _sst_module.SystemPropagator()
dynamics = lambda x, u, t: cpp_propagator.propagate(system, x, u, t)
vis = CartpoleVisualizer(dynamics, params)
states = path
actions = controls
sgs[0] = wrap_angle(sgs[0], system)
sgs[1] = wrap_angle(sgs[1], system)
print('states:')
print(states)
traj = vis.animate(np.array(states), np.array(actions), np.array(costs), obs_list, np.array(sgs), system)
traj = traj[:-1:10].tolist() + [traj[-1]]  # sub sample
traj = np.array(traj)
color_dict = {'state_start_color': 'springgreen', 'state_intermediate_color': 'dodgerblue', 'state_goal_color': 'red',
              'pole_start_color': 'springgreen', 'pole_intermediate_color': 'cornflowerblue', 'pole_goal_color': 'red',
              'cart_start_color': 'cornflowerblue', 'cart_intermediate_color': 'cornflowerblue', 'cart_goal_color': 'cornflowerblue',
              'obstacle_color': 'slategray'}

ani = vis.gen_video(traj, obs_list, np.array(sgs), color_dict, system)
ani.save('cartpole_env%d_path%d_sst.mp4' % (obs_idx, p_idx), fps=50)
#ani.save('cartpole_env%d_path%d_sst.gif' % (obs_idx, p_idx), writer='imagemagick')

