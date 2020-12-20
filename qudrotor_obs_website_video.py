## from ctypes import *
#ctypes.cdll.LoadLibrary('')
#lib1 = CDLL("deps/sparse_rrt/deps/trajopt/build/lib/libsco.so")
#lib2 = CDLL("deps/sparse_rrt/deps/trajopt/build/lib/libutils.so")
import sys
sys.path.append('/media/arclabdl1/HD1/YLmiao/YLmiao_from_unicorn/YLmiao/mpc-mpnet-cuda-yinglong')
sys.path.append('/media/arclabdl1/HD1/YLmiao/YLmiao_from_unicorn/YLmiao/mpc-mpnet-cuda-yinglong/deps/sparse_rrt-1')
#sys.path.append('/media/arclabdl1/HD1/Linjun/mpc-mpnet-py/deps/sparse_rrt-1')
#sys.path.append('/media/arclabdl1/HD1/Linjun/mpc-mpnet-py')
from sparse_rrt import _deep_smp_module
from tqdm.notebook import tqdm

import os

import numpy as np
from mpnet.sst_envs.utils import load_data, get_obs
import pickle
import time
import click
from tqdm.auto import tqdm
from pathlib import Path
import importlib
from matplotlib import pyplot as plt
from sparse_rrt import _deep_smp_module, _sst_module


#[(0, 932), (1, 935), (2, 923), (8, 141), (5,931), (7, 927)]
# (5,931), (6, 286)
from mpnet.sst_envs.utils import load_data, get_obs_3d
system = "quadrotor_obs"

env_id = 5

traj_id = 913
data = load_data(system, env_id, traj_id)
print(data)
cost = data['cost'].sum()
print(cost)

obs_list=get_obs_3d('quadrotor_obs',"obs")[env_id]
obc_list = np.load('/media/arclabdl1/HD1/YLmiao/YLmiao_from_unicorn/YLmiao/mpc-mpnet-cuda-yinglong/mpnet/sst_envs/{}_env_vox.npy'.format(system))
obc_list = obc_list[env_id].reshape(-1)

mpc_path, mpc_control, mpc_time = np.load('/media/arclabdl1/HD1/YLmiao/YLmiao_from_unicorn/YLmiao/mpc-mpnet-cuda-yinglong/results/cpp_full/quadrotor_obs/default/paths/path_%d_%d.npy' % (env_id, traj_id),
               allow_pickle=True)        

print('env_id: ')
print(env_id)
print('traj_id:')
print(traj_id)
        

# Create custom system
#obs_list = [[-10., -3.],
#            [0., 3.],
#            [10, -3.]]

#obc_list = obc_list_total[obs_idx]
print('generated.')
print(obs_list.shape)

start_state = data['start_goal'][0]
goal_state = data['start_goal'][1]



params = {
    'solver_type' : "cem",
    'n_problem': 1,#128,
    'n_sample': 512,#32,g
    'n_elite': 64,#4,
    'n_t': 1,
    'max_it': 100,
    'converge_r': 1e-10,

    'dt': 2e-3,

    'mu_u': np.array([-10., 0., 0., 0.]),
    'sigma_u': np.array([15., 1., 1., 1.]),

    'mu_t': .8,
    'sigma_t': .8,
    't_max': 2.,
    'verbose': False,#True,# 
    'step_size': 0.8,

    "goal_radius": 2.0,

    "sst_delta_near": .1,
    "sst_delta_drain": 0.01,
    "goal_bias": 0.02,

    "width": 1,
    "hybrid": False,
    "hybrid_p": 0.0,

    "cost_samples": 1,
    "mpnet_weight_path":"/media/arclabdl1/HD1/Linjun/mpc-mpnet-py/mpnet/exported/output/quadrotor_obs/mpnet-tree-batch-128.pt",
    #"mpnet_weight_path":"mpnet/exported/output/cartpole_obs/mpnet_10k_external_v2_deep.pt",
    #"mpnet_weight_path":"mpnet/exported/output/cartpole_obs/mpnet_10k.pt",

    # "mpnet_weight_path":"mpnet/exported/output/cartpole_obs/mpnet_10k_nonorm.pt",
    # "mpnet_weight_path":"mpnet/exported/output/cartpole_obs/mpnet_subsample0.5_10k.pt",

    "cost_predictor_weight_path": "/media/arclabdl1/HD1/Linjun/mpc-mpnet-py/mpnet/exported/output/cartpole_obs/cost_10k.pt",
    "cost_to_go_predictor_weight_path": "/media/arclabdl1/HD1/Linjun/mpc-mpnet-py/mpnet/exported/output/cartpole_obs/cost_to_go_10k.pt",

    "refine": False,
    "using_one_step_cost": False,
    "refine_lr": 0,
    "refine_threshold": 0,
    "device_id": "cuda:3",

    "cost_reselection": False,
    "number_of_iterations": 1000,
    "weights_array": np.ones(13)

}



planner = _deep_smp_module.DSSTMPCWrapper(
    system_type='quadrotor_obs',
    solver_type="cem",
    start_state=data['start_goal'][0],
#             goal_state=np.array(ref_path[-1]),
    goal_state=data['start_goal'][1],
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
    obs_voxel_array=obc_list
)

system = _sst_module.Quadrotor()
cpp_propagator = _sst_module.SystemPropagator()
dynamics = lambda x, u, t: cpp_propagator.propagate(system, x, u, t)



from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def draw_line_3d(ax, p, p_index, color='b', alpha=1):
    for p_i in p_index:
        ax.plot3D(p[p_i, 0], p[p_i, 1], p[p_i, 2], c=color, alpha=alpha)

def centered_box_to_points_3d(center, size):
    half_size = [s/2 for s in size]
    direction, p = [1, -1], []
    for x_d in direction:
        for y_d in direction:
            for z_d in direction:
                p.append([center[di] + [x_d, y_d, z_d][di] * half_size[0] for di in range(3)])
    return p

def rot_frame_3d(state):
    frame_size = 0.25
    b, c, d, a = state[3:7]
    rot_mat = np.array([[2 * a**2 - 1 + 2 * b**2, 2 * b * c + 2 * a * d, 2 * b * d - 2 * a * c],
                        [2 * b * c - 2 * a * d, 2 * a**2 - 1 + 2 * c**2, 2 * c * d + 2 * a * b],
                        [2 * b * d + 2 * a * c, 2 * c * d - 2 * a * b, 2 * a**2 - 1 + 2 * d**2]])
    quadrotor_frame = np.array([[frame_size, 0, 0],
                                 [0, frame_size, 0],
                                 [-frame_size, 0, 0],
                                 [0, -frame_size, 0]]).T
    quadrotor_frame = rot_mat @ quadrotor_frame + state[:3].reshape(-1, 1)
    return quadrotor_frame

def q_to_points_3d(state):
    quadrotor_frame = rot_frame_3d(state)   
    max_min, direction = [np.max(quadrotor_frame, axis=1), np.min(quadrotor_frame, axis=1)], [1, 0]
    p = []
    for x_d in direction:
        for y_d in direction:
            for z_d in direction:
                p.append([max_min[x_d][0], max_min[y_d][1], max_min[z_d][2]])
    return np.array(p)

def draw_box_3d(ax, p, color='b', alpha=1, face_alpha=0.1, surface_color='blue', linewidths=1, edgecolors='k'):
    index_lists = [[[0, 4], [4, 6], [6, 2], [2, 0], [0, 1], [1, 5], [5, 7], [7, 3], [3, 1], [1, 5]],
                  [[4, 5]],
                  [[6, 7]],
                  [[2, 3]]]
    for p_i in index_lists:
        draw_line_3d(ax, np.array(p), p_i, color=color, alpha=alpha)
    if surface_color is not None:
        edges = [[p[e_i] for e_i in f_i] for f_i in [[0, 1, 5, 4],
                                                     [4, 5, 7, 6],
                                                     [6, 7, 3, 2],
                                                     [2, 0, 1, 3],
                                                     [2, 0, 4, 6],
                                                     [3, 1, 5, 7]]]
        faces = Poly3DCollection(edges, linewidths=linewidths, edgecolors=edgecolors)
        faces.set_facecolor(surface_color)
        faces.set_alpha(face_alpha)
        ax.add_collection3d(faces)

def visualize_quadrotor_path(path, draw_bbox=True, alpha=1.):
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')
    # draw a ball of the goal region
    """
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    x = 1 * np.outer(np.cos(u), np.sin(v)) * params['goal_radius']
    y = 1 * np.outer(np.sin(u), np.sin(v)) * params['goal_radius']
    z = 1 * np.outer(np.ones(np.size(u)), np.cos(v)) * params['goal_radius']
    # transport the ball to goal position
    x = x + data['start_goal'][1][0]
    y = y + data['start_goal'][1][1]
    z = z + data['start_goal'][1][2]
    ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='r', linewidth=0, alpha=0.3)
    """
    draw_box_3d(ax, centered_box_to_points_3d([0, 0, 0], [10, 10, 10]), alpha=1, color='slategray', surface_color=None)

    if path is not None:
#         ax.scatter(path[:, 0], path[:, 1], path[:, 2], c='dodgerblue', alpha=alpha, linestyle='dotted')
#         ax.plot(path[:, 0], path[:, 1], path[:, 2], c='dodgerblue', alpha=alpha, linestyle='dotted')
#         ax.scatter(path[:, 0], path[:, 1], path[:, 2], c='dodgerblue', alpha=alpha)
        ax.plot(path[:, 0], path[:, 1], path[:, 2], c='dodgerblue', alpha=alpha)
        ax.scatter(start_state[0], start_state[1], start_state[2], c='springgreen', s=200)
        ax.scatter(goal_state[0], goal_state[1], goal_state[2], c='red', marker='*', s=200)
        for waypoint in path:
            f = rot_frame_3d(waypoint)
            ax.scatter(f[0], f[1], f[2], color='red', s=60, alpha=0.3)
            ax.plot(f[0,[0, 2]], f[1, [0, 2]], f[2, [0, 2]], c='dodgerblue')
            ax.plot(f[0,[1, 3]], f[1, [1, 3]], f[2, [1, 3]], c='dodgerblue')

            if draw_bbox:
                draw_box_3d(ax, q_to_points_3d(waypoint), alpha=0.1, face_alpha=0.1, color='dodgerblue', surface_color="dodgerblue", linewidths=0.)
    for obs in obs_list:
        draw_box_3d(ax, centered_box_to_points_3d(center=obs, size=[params['width']]*3),\
                    surface_color="slategray", face_alpha=1.)

    ax.set_xlim3d(-5, 5)
    ax.set_ylim3d(-5, 5)
    ax.set_zlim3d(-5, 5)
    plt.axis('off')
    ax.grid(b=None)
    return fig, ax


def propagate_path(state, control=None, time=None):
    dense_path = [state[0]]
    for i in range(len(state)-1):
        if control is None:
            solution_u, solution_t = planner.steer_solution(state[i], state[i+1])
        else:
            solution_u, solution_t = [control[i]], [time[i]]
        # proapgate
        s = state[i]
        for j in range(len(solution_u)):
            action = solution_u[j]
            num_steps = int(np.round(solution_t[j]/params['dt']))
            for k in range(num_steps):
                s = dynamics(s, action, params['dt'])
                dense_path.append(np.array(s))
                
    dense_path = np.array(dense_path)
    return dense_path[::100]


def animate_init(traj, draw_bbox=True, alpha=1.0):
    pass
def animate(traj, draw_bbox=True, alpha=1.):
    pass


class QuadrotorVisualizer():
    def __init__(self, system, params):
        self.dt = 2
        self.fig = plt.figure(figsize=(20,20))
        ax = fig.add_subplot(111, projection='3d')
    def _init(self):
        ##### handle the animation
        # clear the current ax
        print("in init")
        ax = self.ax
        ax.clear()
        # add goal
        goal_state = self.sgs[1]
        ax.scatter(goal_state[0], goal_state[1], goal_state[2], c='red', marker='*', s=200)
        ax.plot(path[:, 0], path[:, 1], path[:, 2], c='dodgerblue', alpha=alpha)
        ax.scatter(start_state[0], start_state[1], start_state[2], c='springgreen', s=200)
        ax.scatter(goal_state[0], goal_state[1], goal_state[2], c='red', marker='*', s=200)
        
        f = rot_frame_3d(start_state)
        self.frame_scat = ax.scatter(f[0], f[1], f[2], color='red', s=60, alpha=0.3)
        self.frame_l1 = ax.plot(f[0,[0, 2]], f[1, [0, 2]], f[2, [0, 2]], c='dodgerblue')[0]
        self.frame_l2 = ax.plot(f[0,[1, 3]], f[1, [1, 3]], f[2, [1, 3]], c='dodgerblue')[0]

        if draw_bbox:
            draw_box_3d(ax, q_to_points_3d(waypoint), alpha=0.1, face_alpha=0.1, color='dodgerblue', surface_color="dodgerblue", linewidths=0.)
        for obs in obs_list:
            draw_box_3d(ax, centered_box_to_points_3d(center=obs, size=[params['width']]*3),\
                        surface_color="slategray", face_alpha=1.)

        ax.set_xlim3d(-5, 5)
        ax.set_ylim3d(-5, 5)
        ax.set_zlim3d(-5, 5)
        plt.axis('off')
        ax.grid(b=None)

        
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
        ax.set_xlim(-25, 25)
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
        ax.set_xlim(-25, 25)
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

        ax.set_xlim(-25, 25)
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
        





# %matplotlib notebook
print('SST: env_id = %d, traj_id = %d' % (env_id, traj_id))
dense_path = propagate_path(data['path'], data['control'], data['cost'])
visualize_quadrotor_path(dense_path, True, 0.5)#,draw_bbox=False)
plt.savefig('quadrotor_sst_obs_%d_p_%d.pdf' % (env_id, traj_id), bbox_inches='tight')



# %matplotlib notebook
print('MPNetTree: env_id = %d, traj_id = %d' % (env_id, traj_id))
print(data['path'].shape)
print(mpc_path.shape)
dense_path = propagate_path(mpc_path)

visualize_quadrotor_path(dense_path)
plt.savefig('quadrotor_mpnet_obs_%d_p_%d.pdf' % (env_id, traj_id), bbox_inches='tight')