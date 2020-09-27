## from ctypes import *
#ctypes.cdll.LoadLibrary('')
#lib1 = CDLL("deps/sparse_rrt/deps/trajopt/build/lib/libsco.so")
#lib2 = CDLL("deps/sparse_rrt/deps/trajopt/build/lib/libutils.so")

import sys
sys.path.append('/media/arclabdl1/HD1/YLmiao/mpc-mpnet-cuda-yinglong/deps/sparse_rrt-1')
sys.path.append('.')

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
import matplotlib
matplotlib.use("Agg")
obs_list = []
LENGTH = 20.
width = 8.
near = width * 1.2
# convert from obs to point cloud
# load generated point cloud
obs_list_total = []
obc_list_total = []
for i in range(10):
    file = open('/media/arclabdl1/HD1/YLmiao/data/kinodynamic/car_obs/obs_%d.pkl' % (i), 'rb')
    obs_list_total.append(pickle.load(file))
    file = open('/media/arclabdl1/HD1/YLmiao/data/kinodynamic/car_obs/obc_%d.pkl' % (i), 'rb')
    obc_list_total.append(pickle.load(file))

#[(0, 932), (1, 935), (2, 923), (8, 141), (5,931), (7, 927)]
# (5,931), (6, 286)

for video_i in range(20):
    obs_idx = np.random.randint(low=5,high=10)
    p_idx = np.random.randint(low=800,high=1000)
    #p_idx = 900
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


    params = {
        'solver_type' : "cem",
        'n_sample': 1024,
        'n_elite': 128,
        'n_t': 2,
        'max_it': 200,
        'converge_r': 1e-10,

        'dt': 2e-3,

        'mu_u': np.array([1., 1.]) ,
        'sigma_u': np.array([1., 0.5]),

        'mu_t': .5,
        'sigma_t': .5,
        't_max': 1.,

        'verbose': False,#True,# 
        'step_size': 0.8,

        "goal_radius": 2.0,

        "sst_delta_near": .001,
        "sst_delta_drain": .0005,
        "goal_bias": 0.08,

        "width": 8,
        "hybrid": False,
        "hybrid_p": 0.0,

        "cost_samples": 1,
        "mpnet_weight_path":"/home/arclabdl1/YLmiao/kinodynamics/mpc-mpnet-cuda-yinglong/mpnet/exported/output/car_obs/mpnet_10k_external_small_model.pt",
        #"mpnet_weight_path":"mpnet/exported/output/cartpole_obs/mpnet_10k_external_v2_deep.pt",
        #"mpnet_weight_path":"mpnet/exported/output/cartpole_obs/mpnet_10k.pt",

        # "mpnet_weight_path":"mpnet/exported/output/cartpole_obs/mpnet_10k_nonorm.pt",
        # "mpnet_weight_path":"mpnet/exported/output/cartpole_obs/mpnet_subsample0.5_10k.pt",

        "cost_predictor_weight_path": "/home/arclabdl1/YLmiao/kinodynamics/mpc-mpnet-cuda-yinglong/mpnet/exported/output/cartpole_obs/cost_10k.pt",
        "cost_to_go_predictor_weight_path": "/home/arclabdl1/YLmiao/kinodynamics/mpc-mpnet-cuda-yinglong/mpnet/exported/output/cartpole_obs/cost_to_go_10k.pt",

        "refine": False,
        "using_one_step_cost": False,
        "refine_lr": 0,
        "refine_threshold": 0,
        "device_id": "cuda:2",

        "cost_reselection": False,
        "number_of_iterations": 5000,
        "weights_array": [1, 1.0, .5],


    }


    # load path
    path = np.load('/media/arclabdl1/HD1/YLmiao/mpc-mpnet-cuda-yinglong/results/cpp_full/car_obs/default/paths/path_%d_%d.npy' % (obs_idx, p_idx), allow_pickle=True)[0]

    sgs = open('/media/arclabdl1/HD1/YLmiao/data/kinodynamic/car_obs/%d/start_goal_%d.pkl' % (obs_idx, p_idx), 'rb')
    sgs = pickle.load(sgs)


    planner = _deep_smp_module.DSSTMPCWrapper(
        system_type='car_obs',
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
        #np=params['n_problem'], 
        ns=params['n_sample'], nt=params['n_t'], ne=params['n_elite'], max_it=params['max_it'],
        converge_r=params['converge_r'], mu_u=params['mu_u'], std_u=params['sigma_u'], mu_t=params['mu_t'], 
        std_t=params['sigma_t'], t_max=params['t_max'], step_size=params['step_size'], integration_step=params['dt'], 
        device_id=params['device_id'], refine_lr=params['refine_lr'],
        weights_array=params['weights_array'],
        obs_voxel_array=obc_list.reshape(-1)
    )



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

    def IsInCollision(x, obc, obc_width=8.):
        car_width = 2.0
        car_len = 1.0
        width = 8.0
        WIDTH = car_width
        LENGTH = car_len
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
                print('state: %d, remaining: %d' % (i, len(states)-i))
                # connect from this state to next
                solution_u, solution_t = planner.steer_solution(states[i], states[i+1])
                #print('solution_u:')
                #print(solution_u)
                #print('solution_t:')
                #print(solution_t)

                #print('start:')
                #print(states[i])
                #print('s:')
                #print(s)
                s = states[i]

                for j in range(len(solution_u)):
                    action = solution_u[j]
                    num_steps = int(np.round(solution_t[j]/self.params['integration_step']))
                    for k in range(num_steps):
                        traj.append(np.array(s))
                        print("porpagating... j = %d, k = %d" % (j, k))
                        #print(s)
                        #print('st:')
                        #print(sT)
                        s = self.system(s, action, self.params['integration_step'])
                        if IsInCollision(s, obs_i):
                            print('collision state:')
                            print(s)
                            print('st:')
                            print(states[i+1])
                        assert not IsInCollision(s, obs_i)
                        s = enforce_bounds(s)

            traj = np.array(traj)
            print("animating...")
            # animate
            self.states = traj
            #print(traj)
            self.obs = obstacles
            print(len(self.states))
            self.total = len(self.states)
            ani = animation.FuncAnimation(self.fig, self._animate, range(0, len(self.states), 1),
                                          interval=self.dt, blit=True, init_func=self._init,
                                          repeat=True)
            return ani




    params = {}
    car_width = 2.0
    car_len = 1.0
    width = 8.0
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
    states = path
    sgs[0] = wrap_angle(sgs[0], propagate_system)
    sgs[1] = wrap_angle(sgs[1], propagate_system)
    #print('states:')
    #print(states)
    anim = vis.animate(np.array(states), None, None, obs_list)
    #HTML(anim.to_html5_video())
    anim.save('car_obs_MPC_env%d_path%d.mp4' % (obs_idx, p_idx))