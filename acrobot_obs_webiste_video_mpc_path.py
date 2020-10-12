## from ctypes import *
#ctypes.cdll.LoadLibrary('')
#lib1 = CDLL("deps/sparse_rrt/deps/trajopt/build/lib/libsco.so")
#lib2 = CDLL("deps/sparse_rrt/deps/trajopt/build/lib/libutils.so")

import sys
sys.path.append('/media/arclabdl1/HD1/YLmiao/YLmiao_from_unicorn/YLmiao/mpc-mpnet-cuda-yinglong/deps/sparse_rrt-1')
#sys.path.append('.')

from sparse_rrt.planners import SST
#from env.cartpole_obs import CartPoleObs
#from env.cartpole import CartPole
#from sparse_rrt.systems import standard_cpp_systems
from sparse_rrt import _sst_module
import numpy as np
import time
from plan_utility.line_line_cc import *
import pickle
from sparse_rrt import _deep_smp_module, _sst_module
import os
obs_list = []
LENGTH = 20.
width = 6.
near = width * 1.2
# convert from obs to point cloud
# load generated point cloud
obs_list_total = []
obc_list_total = []
for i in range(10):
    file = open('/media/arclabdl1/HD1/YLmiao/YLmiao_from_unicorn/YLmiao/data/kinodynamic/acrobot_obs_backup/obs_%d.pkl' % (i), 'rb')
    obs_list_total.append(pickle.load(file))
    file = open('/media/arclabdl1/HD1/YLmiao/YLmiao_from_unicorn/YLmiao/data/kinodynamic/acrobot_obs_backup/obc_%d.pkl' % (i), 'rb')
    obc_list_total.append(pickle.load(file))

#[(0, 932), (1, 935), (2, 923), (8, 141), (5,931), (7, 927)]
# (5,931), (6, 286)

obs_idx = 1
p_idx = 916
        
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



params = {
        'n_problem': 1,
        'n_sample': 1024,
        'n_elite': 128,
        'n_t': 1,
        'max_it': 1000,
        #'converge_r': 5e-1,
        'converge_r': 1e-10,
    
        'dt': 2e-2,
        'mu_u': [0],
        'sigma_u': [6],
         'mu_t': 2e-1,
         'sigma_t': 8e-1,
         't_max': 2.,
#         'mu_t': 2e-1,
#         'sigma_t': 2e-1,
#         't_max': .5,


        'verbose': False,# False,#
        'step_size': 0,

        "goal_radius": 2.0,

        "sst_delta_near": 1,
        "sst_delta_drain": 1e-1,
        "goal_bias": 0.04,

        "width": 6,
        "hybrid": False,#True,# 
        "hybrid_p": 0.01,
        
        "min_time_steps": 5,
        "max_time_steps": 100,
    
        "cost_samples": 1,
        "mpnet_weight_path":"/media/arclabdl1/HD1/YLmiao/YLmiao_from_unicorn/YLmiao/mpc-mpnet-cuda-yinglong/mpnet/exported/output/acrobot_obs/mpnet_10k.pt",
        #"mpnet_weight_path":"mpnet/exported/output/cartpole_obs/mpnet_10k_external_v3_multigoal.pt",
        #"mpnet_weight_path":"mpnet/exported/output/cartpole_obs/mpnet_10k_external_v2_deep.pt",
#         "mpnet_weight_path":"mpnet/exported/output/cartpole_obs/mpnet_10k.pt",
#         "mpnet_weight_path":"mpnet/exported/output/cartpole_obs/mpnet_10k_branch.pt",


        # "mpnet_weight_path":"mpnet/exported/output/cartpole_obs/mpnet_10k_nonorm.pt",
        # "mpnet_weight_path":"mpnet/exported/output/cartpole_obs/mpnet_subsample0.5_10k.pt",

        "cost_predictor_weight_path": "/media/arclabdl1/HD1/YLmiao/YLmiao_from_unicorn/YLmiao/mpc-mpnet-cuda-yinglong/mpnet/exported/output/cartpole_obs/cost_10k.pt",
        "cost_to_go_predictor_weight_path": "/media/arclabdl1/HD1/YLmiao/YLmiao_from_unicorn/YLmiao/mpc-mpnet-cuda-yinglong/mpnet/exported/output/cartpole_obs/cost_to_go_10k.pt",

        "refine": False,
        "using_one_step_cost": False,
        "refine_lr": 0,
        "refine_threshold": 0,
        "device_id": "cuda:3",

        "cost_reselection": False,
        "number_of_iterations": 500,
        "weights_array": [1, 1, 1, 1],

    }

path_filename = '/media/arclabdl1/HD1/Linjun/mpc-mpnet-py/results/traj/acrobot_obs/shm/%d/env_%d_id_%d.npy' % (obs_idx, obs_idx, p_idx)
path = np.load(path_filename, allow_pickle=True)
print(path)
sgs = open('/media/arclabdl1/HD1/YLmiao/YLmiao_from_unicorn/YLmiao/data/kinodynamic/acrobot_obs_backup/%d/start_goal_%d.pkl' % (obs_idx, p_idx), 'rb')
sgs = pickle.load(sgs)



planner = _deep_smp_module.DSSTMPCWrapper(
    system_type='acrobot_obs',
    solver_type="cem",
    start_state=np.array(path[0]),
#             goal_state=np.array(ref_path[-1]),
    goal_state=np.array(sgs[-1]),
    goal_radius=params['goal_radius'],
    random_seed=0,
    sst_delta_near=params['sst_delta_near'],
    sst_delta_drain=params['sst_delta_drain'],
    obs_list=obs_list,
    width=params['width'],
    verbose=params['verbose'],
    mpnet_weight_path=params['mpnet_weight_path'], 
    cost_predictor_weight_path=params['cost_predictor_weight_path'],
    cost_to_go_predictor_weight_path=params['cost_to_go_predictor_weight_path'],
    num_sample=params['cost_samples'],
    np=params['n_problem'], ns=params['n_sample'], nt=params['n_t'], ne=params['n_elite'], max_it=params['max_it'],
    converge_r=params['converge_r'], mu_u=params['mu_u'], std_u=params['sigma_u'], mu_t=params['mu_t'], 
    std_t=params['sigma_t'], t_max=params['t_max'], step_size=params['step_size'], integration_step=params['dt'], 
    device_id=params['device_id'], refine_lr=params['refine_lr'],
    weights_array=params['weights_array'],
    obs_voxel_array=obc_list.reshape(-1)
)




from plan_utility.line_line_cc import line_line_cc

def IsInCollision(x, obc, obc_width=6.):
    STATE_THETA_1, STATE_THETA_2, STATE_V_1, STATE_V_2 = 0, 1, 2, 3
    MIN_V_1, MAX_V_1 = -6., 6.
    MIN_V_2, MAX_V_2 = -6., 6.
    MIN_TORQUE, MAX_TORQUE = -4., 4.

    MIN_ANGLE, MAX_ANGLE = -np.pi, np.pi

    LENGTH = 20.
    m = 1.0
    lc = 0.5
    lc2 = 0.25
    l2 = 1.
    I1 = 0.2
    I2 = 1.0
    l = 1.0
    g = 9.81
    pole_x0 = 0.
    pole_y0 = 0.
    pole_x1 = LENGTH * np.cos(x[STATE_THETA_1] - np.pi / 2)
    pole_y1 = LENGTH * np.sin(x[STATE_THETA_1] - np.pi / 2)
    pole_x2 = pole_x1 + LENGTH * np.cos(x[STATE_THETA_1] + x[STATE_THETA_2] - np.pi / 2)
    pole_y2 = pole_y1 + LENGTH * np.sin(x[STATE_THETA_1] + x[STATE_THETA_2] - np.pi / 2)
    for i in range(len(obc)):
        for j in range(0, 8, 2):
            x1 = obc[i][j]
            y1 = obc[i][j+1]
            x2 = obc[i][(j+2) % 8]
            y2 = obc[i][(j+3) % 8]
            if line_line_cc(pole_x0, pole_y0, pole_x1, pole_y1, x1, y1, x2, y2):
                return True
            if line_line_cc(pole_x1, pole_y1, pole_x2, pole_y2, x1, y1, x2, y2):
                return True
    return False


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







import matplotlib
matplotlib.use('Agg')
from visual.visualizer import Visualizer
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
class AcrobotVisualizer(Visualizer):
    def __init__(self, system, params, color_dict={'obs_color': 'black'}):
        super(AcrobotVisualizer, self).__init__(system, params)
        self.dt = 20
        self.fig = plt.gcf()
        self.fig.set_figheight(5)
        self.fig.set_figwidth(10)
        self.ax1 = plt.subplot(121)
        self.ax2 = plt.subplot(122)
        self.color_dict = color_dict

    def _state_to_xy(self, state):
        angle0 = state[0]
        angle1 = state[1]
        x0 = 0.
        y0 = 0.
        x1 = LENGTH * np.cos(angle0 - np.pi/2)
        y1 = LENGTH * np.sin(angle0 - np.pi/2)
        x2 = x1 + LENGTH * np.cos(angle0 + angle1 - np.pi/2)
        y2 = y1 + LENGTH * np.sin(angle0 + angle1 - np.pi/2)
        return x0, y0, x1, y1, x2, y2
    def _init(self):
        ##### handle the animation
        # clear the current ax
        print("in init")
        ax = self.ax1
        ax.clear()
        # add patches
        goal_state = self.sgs[1]
        x0, y0, x1, y1, x2, y2 = self._state_to_xy(goal_state)
        print('x1: %f, y1: %f, x2: %f, y2: %f' % (x1, y1, x2, y2))
        self.l1_goal = ax.plot([x0,x1,x2], [y0,y1,y2], c=self.color_dict['goal_color'])[0]

        
        state = self.states[0]
        print('state:')
        print(state)
        x0, y0, x1, y1, x2, y2 = self._state_to_xy(state)
        print('x1: %f, y1: %f, x2: %f, y2: %f' % (x1, y1, x2, y2))
        self.l1 = ax.plot([x0,x1,x2], [y0,y1,y2], c=self.color_dict['intermediate_color'])[0]
        self.recs = []
        for i in range(len(self.obs)):
            x, y = self.obs[i]
            obs = patches.Rectangle((x-self.params['obs_w']/2,y-params['obs_h']/2),\
                                       self.params['obs_w'],self.params['obs_h'],\
                                      linewidth=.5,edgecolor=self.color_dict['obs_color'],facecolor=self.color_dict['obs_color'])
            self.recs.append(obs)
            ax.add_patch(obs)
        self.recs.append(self.l1)

        #### handle search space
        ax = self.ax2
        ax.clear()
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(-np.pi, np.pi)


        dtheta = 0.1
        feasible_points = []
        infeasible_points = []
        imin = 0
        imax = int(2*np.pi/dtheta)


        for i in range(imin, imax):
            for j in range(imin, imax):
                x = np.array([dtheta*i-np.pi, dtheta*j-np.pi, 0., 0.])
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
        #scat_feas =ax.scatter(feasible_points[:,0], feasible_points[:,1], c='white')
        scat_infeas = ax.scatter(infeasible_points[:,0], infeasible_points[:,1], c=self.color_dict['obs_color'])

        #self.recs.append(scat_feas)
        self.recs.append(scat_infeas)

        
        goal_scat_state = ax.scatter(goal_state[0], goal_state[1], c=color_dict['goal_color'], s=25.0)

        
        scat_state = ax.scatter(state[0], state[1], c=color_dict['intermediate_color'], s=25.0)
        self.recs.append(scat_state)
        print("after init")

        return self.recs
    def _animate(self, i):
        print('animating, frame %d/%d' % (i, self.total))

        ax = self.ax1
        ax.set_xlim(-40, 40)
        ax.set_ylim(-40, 40)
        state = self.states[i]
        x0, y0, x1, y1, x2, y2 = self._state_to_xy(state)
        self.l1.set_xdata([x0,x1,x2])
        self.l1.set_ydata([y0,y1,y2])

        # handle search space
        ax = self.ax2
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(-np.pi, np.pi)
        self.recs[-1].set_offsets([state[0], state[1]])
        # print location of point
        return self.recs



    def animate(self, states, actions, costs, obstacles, sg):
        '''
        given a list of states, actions and obstacles, animate the robot
        '''

        new_obs_i = []
        obs_width = 6.0
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
            s = states[i]
            print('state: %d, remaining: %d' % (i, len(states)-i))
            #action = actions[i]
            
            # connect from this state to next
            solution_u, solution_t = planner.steer_solution(states[i], states[i+1])
            print('solution_u:')
            print(solution_u)
            print('solution_t:')
            print(solution_t)
            for j in range(len(solution_u)):
                action = solution_u[j]
                num_steps = int(np.round(solution_t[j]/self.params['integration_step']))

                for k in range(num_steps):
                    traj.append(np.array(s))
                    #print("porpagating...")
                    #print(s)
                    #print('st:')
                    #print(sT)
                    s = self.system(s, action, self.params['integration_step'])
                    assert not IsInCollision(s, obs_i)
            print('after steering state: ', s)
            print('next state: ', states[i+1])
        self.states = traj
        self.obs = obstacles

        return np.array(traj)

    # plot the trajectory
    def plot(self, traj, obstacles, sg, color_dict, fig):
        self.fig = fig
        self.fig.set_figheight(5)
        self.fig.set_figwidth(10)
        self.ax1 = fig.add_subplot(121)
        self.ax2 = fig.add_subplot(122)

        traj = np.array(traj)
        self.obs = obstacles

        print("animating...")
        # animate
        self.states = traj
        #self.obs = obstacles
        print(len(self.states))
        self.total = len(self.states)
        self._init()
        
        to_plot_list_x = []
        to_plot_list_y = []
        for i in range(len(traj)):
            if i % 10 == 0:
                x0, y0, x1, y1, x2, y2 = self._state_to_xy(traj[i])
                ax = self.ax1
                ax.set_xlim(-40, 40)
                ax.set_ylim(-40, 40)
                #if i == 0:
                #    ax.plot([x0,x1,x2],[y0,y1,y2],alpha=1, c='green')
                #else:
                #    ax.plot([x0,x1,x2],[y0,y1,y2],alpha=float(i)/len(traj), c='blue')
                to_plot_list_x.append([x0,x1,x2])
                to_plot_list_y.append([y0,y1,y2])
            ax = self.ax2
            scat_state = ax.scatter(traj[i][0], traj[i][1], c=color_dict['intermediate_color'], s=25.0)
        #colors = ['green', 'red']
        colors = [color_dict['start_color'], color_dict['goal_color']]
        ax = self.ax1
        cm = LinearSegmentedColormap.from_list("Custom", colors, N=len(to_plot_list_x))
        for i in range(len(to_plot_list_x)):
            #ax.plot(to_plot_list_x[i], to_plot_list_y[i], alpha=float(i)/len(to_plot_list_x), c=cm(float(i)/len(to_plot_list_x)))
            if i == 0:
                ax.plot(to_plot_list_x[i], to_plot_list_y[i], alpha=1, c=color_dict['start_color'])
            else:
                ax.plot(to_plot_list_x[i], to_plot_list_y[i], alpha=float(i)/len(to_plot_list_x), c=color_dict['intermediate_color'])

            
        # start and goal
        x0, y0, x1, y1, x2, y2 = self._state_to_xy(sg[1])
        ax = self.ax1
        ax.set_xlim(-40, 40)
        ax.set_ylim(-40, 40)
        ax.plot([x0,x1,x2],[y0,y1,y2],alpha=1, c=color_dict['goal_color'])
        ax.axis('off')

        ax = self.ax2
        scat_state = ax.scatter(sg[0][0], sg[0][1], c=color_dict['start_color'], s=50.0)
        scat_state = ax.scatter(sg[1][0], sg[1][1], c=color_dict['goal_color'], s=50.0, marker='*')
        ax.axis('off')
        return self.fig
        #plt.savefig("acrobot_mpnettree_obs_%d_p_%d.png" % (obs_idx, p_idx), bbox_inches='tight')
        #plt.show()
        
    def gen_video(self, traj, obs_list, sgs, color_dict, system):
        self.states = traj
        self.obs_list = obs_list
        self.color_dict = color_dict
        self.total = len(self.states)
        self.sgs = sgs

        ani = animation.FuncAnimation(self.fig, self._animate, range(0, len(self.states)),
                                      interval=self.dt, blit=True, init_func=self._init,
                                      repeat=True)
        return ani
        
        
params = {}
params['obs_w'] = width
params['obs_h'] = width
params['integration_step'] = 0.02
system = _sst_module.TwoLinkAcrobot()
params['wrap_angle_system'] = system

cpp_propagator = _sst_module.SystemPropagator()
dynamics = lambda x, u, t: cpp_propagator.propagate(system, x, u, t)

vis = AcrobotVisualizer(dynamics, params)
states = path
sgs[0] = wrap_angle(sgs[0], system)
sgs[1] = wrap_angle(sgs[1], system)
print('states:')
print(states)
traj = vis.animate(np.array(states), None, None, obs_list, np.array(sgs))
#HTML(anim.to_html5_video())
#anim.save('acrobot_env%d_path%d.mp4' % (obs_idx, p_idx))

color_dict = {'start_color': 'springgreen', 'intermediate_color': 'dodgerblue', 'goal_color': 'red', 'obs_color':'slategray'}


ani = vis.gen_video(traj, obs_list, np.array(sgs), color_dict, system)
ani.save('acrobot_env%d_path%d_mpc_path.mp4' % (obs_idx, p_idx))
ani.save('acrobot_env%d_path%d_mpc_path.gif' % (obs_idx, p_idx), writer='imagemagick')

