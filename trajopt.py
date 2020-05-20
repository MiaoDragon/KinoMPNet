"""
This implements the trajopt method for collision avoidance given initial trajectory guess
input:
    trajectory:
        (assuming time is 0.02s)
        state:
        control:
"""

import torch
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
import pickle
import acrobot
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
num_dis_pts = 10 # num of points sampled on the acrobot for each link
dt = 0.02

def dynamics(state, control):
    '''
    Port of the cpp implementation for computing state space derivatives
    '''
    theta2 = state[:,STATE_THETA_2]
    theta1 = state[:,STATE_THETA_1] - np.pi/2
    theta1dot = state[:,STATE_V_1]
    theta2dot = state[:,STATE_V_2]
    _tau = control[:,0]
    d11 = m * lc2 + m * (l2 + lc2 + 2 * l * lc * torch.cos(theta2)) + I1 + I2
    d22 = m * lc2 + I2
    d12 = m * (lc2 + l * lc * torch.cos(theta2)) + I2
    d21 = d12

    c1 = -m * l * lc * theta2dot * theta2dot * torch.sin(theta2) - (2 * m * l * lc * theta1dot * theta2dot * torch.sin(theta2))
    c2 = m * l * lc * theta1dot * theta1dot * torch.sin(theta2)
    g1 = (m * lc + m * l) * g * torch.cos(theta1) + (m * lc * g * torch.cos(theta1 + theta2))
    g2 = m * lc * g * torch.cos(theta1 + theta2)

    u2 = _tau - 1 * .1 * theta2dot
    u1 = -1 * .1 * theta1dot
    theta1dot_dot = (d22 * (u1 - c1 - g1) - d12 * (u2 - c2 - g2)) / (d11 * d22 - d12 * d21)
    theta2dot_dot = (d11 * (u2 - c2 - g2) - d21 * (u1 - c1 - g1)) / (d11 * d22 - d12 * d21)
    deriv = torch.stack([theta1dot, theta2dot, theta1dot_dot, theta2dot_dot]).t()
    return deriv

def distance(point1, point2):
    LENGTH = 20.
    x = torch.cos(point1[0] - np.pi / 2)+torch.cos(point1[0] + point1[1] - np.pi / 2)
    y = torch.sin(point1[0] - np.pi / 2)+torch.sin(point1[0] + point1[1] - np.pi / 2)
    x2 = torch.cos(point2[0] - np.pi / 2)+torch.cos(point2[0] + point2[1] - np.pi / 2)
    y2 = torch.sin(point2[0] - np.pi / 2)+torch.sin(point2[0] + point2[1] - np.pi / 2)
    return LENGTH*LENGTH*((x-x2)**2+(y-y2)**2)



def collision_loss(obs, state):
    # given the obs and state, compute the collision loss
    x1 = torch.cos(state[:,0] - np.pi / 2)
    y1 = torch.sin(state[:,0] - np.pi / 2)
    dx = torch.cos(state[:,0] + state[:,1] - np.pi / 2)
    dx = torch.sin(state[:,0] + state[:,1] - np.pi / 2)
    loss = 0.
    collision_K = 10.  # parameter for collision 20?
    for i in range(1,num_dis_pts+1):
        loss_i = (x1*i*LENGTH/num_dis_pts - obs[0])**2 + (y1*i*LENGTH/num_dis_pts - obs[1])**2
        loss_i = torch.sum(1. / loss_i)
        loss += loss_i

        loss_i = (x1 + dx*i*LENGTH/num_dis_pts - obs[0])**2 + (y1 + dx*i*LENGTH/num_dis_pts - obs[1])**2
        loss_i = torch.sum(1. / loss_i)
        loss += loss_i
    loss = loss / num_dis_pts * collision_K
    return loss
def dynamics_loss(start, start_control, state, control):
    #for i in range(len(state)-1):
    #    #pass
    #    print(state[i] + dynamics(state[i].unsqueeze(0), control[i].unsqueeze(0))*dt - state[i+1])
    state_dot = dynamics(state[:-1], control)
    loss = torch.sum((state_dot*dt + state[:-1] - state[1:])**2)
    dynamics_K = 1.
    # for start
    loss += torch.sum((start+dt*dynamics(start, start_control)-state[1].unsqueeze(0))**2)
    loss = loss * dynamics_K
    return loss
def trajopt(init_state, init_control, obs, lr ,opt):
    max_iter = 10000
    #max_iter = 100
    state = torch.from_numpy(init_state[1:]).clone()
    control = torch.from_numpy(init_control[1:]).clone()
    start = torch.from_numpy(init_state[0]).clone().unsqueeze(0)  # start is not optimized
    start_control = torch.from_numpy(init_control[0]).clone().unsqueeze(0)
    goal = torch.from_numpy(init_state[-1]).clone().detach()
    obs = torch.from_numpy(obs).clone().detach()

    state = Variable(state, requires_grad=True)
    control = Variable(control, requires_grad=True)
    if torch.cuda.is_available():
        state.cuda()
        control.cuda()
        start.cuda()
        start_control.cuda()
        goal.cuda()
        obs.cuda()
    optimizer = opt([state, control], lr=lr, momentum=0.9)
    for i in range(max_iter):
        optimizer.zero_grad()
        c_loss = 0.
        d_loss = 0.
        f_loss = 0.
        for obs_i in range(len(obs)):
            c_loss += collision_loss(obs[obs_i], state)
        d_loss = dynamics_loss(start, start_control, state, control)# / len(state) * 10
        #print('state[-1]:')
        #print(state[-1])
        #print('goal:')
        #print(goal)
        #print('diff:')
        #print(state[-1,:2] - goal[:2])
        f_loss = torch.sum((state[-1,:2] - goal[:2])**2)  # loss for endpoint
        loss = c_loss + d_loss + f_loss
        loss.backward()
        optimizer.step()
        print('iteration %d' % (i))
        print('collision loss: %f' % (c_loss))
        print('dynamics loss: %f' % (d_loss))
        print('endpoint loss: %f' % (f_loss))
        print('total loss: %f' % (loss))

    out_path = [init_state[0]]
    for i in range(len(state)):
        out_path.append(state[i].cpu().data.numpy())
    out_control = [init_control[0]]
    for i in range(len(control)):
        out_control.append(control[i].cpu().data.numpy())
    return out_path, out_control, c_loss.cpu().data.item(), d_loss.cpu().data.item(), f_loss.cpu().data.item()
obs_list_total = []
obc_list_total = []
for i in range(2):
    file = open('data/acrobot_simple/obs_%d.pkl' % (i), 'rb')
    obs_list_total.append(pickle.load(file))
    file = open('data/acrobot_simple/obc_%d.pkl' % (i), 'rb')
    obc_list_total.append(pickle.load(file))

obs_idx = 1
p_idx =4
# Create custom system
#obs_list = [[-10., -3.],
#            [0., 3.],
#            [10, -3.]]
obs_list_total[1][0][0] += 3.
obs_list_total[1][0][1] -= 3.

obs_list_total = np.array(obs_list_total)
obc_list_total = np.array(obc_list_total)
obs_list = obs_list_total[obs_idx]
obc_list = obc_list_total[obs_idx]
print('generated.')
print(obs_list.shape)


path = open('data/acrobot_simple/%d/path_%d.pkl' % (obs_idx, p_idx), 'rb')
path = pickle.load(path)
controls = open('data/acrobot_simple/%d/control_%d.pkl' % (obs_idx, p_idx), 'rb')
controls = pickle.load(controls)
costs = open('data/acrobot_simple/%d/cost_%d.pkl' % (obs_idx, p_idx), 'rb')
costs = pickle.load(costs)
path = path[6:8]
controls = controls[6:8]
costs = costs[6:8]


init_path = [path[0]]
init_control = []
init_cost = []
system = acrobot.Acrobot()
# obtain propagated trajectory
for i in range(len(costs)):
    num_steps = int(costs[i] / dt)
    for j in range(num_steps):
        path_i = system.propagate(init_path[-1], controls[i], 1, dt)
        
        #print(dynamics(torch.from_numpy(init_path[-1]).unsqueeze(0),torch.from_numpy(controls[i]).unsqueeze(0))*dt+torch.from_numpy(init_path[-1])-torch.from_numpy(path_i))     
        
        init_path.append(path_i)
        init_control.append(controls[i])
        init_cost.append(dt)
init_path = np.array(init_path)
init_control = np.array(init_control)
init_cost = np.array(init_cost)
file = open('trajopt_init_path.pkl', 'wb')
pickle.dump(init_path, file)
file = open('trajopt_init_control.pkl', 'wb')
pickle.dump(init_control, file)
file.close()

init_path = init_path[30:60]
init_control = init_control[30:59]

for lr in [0.001, 0.005, 0.01]: #0.05, 0.1]:#, 0.2, 0.25]:
    for opt_name in ['SGD', 'RMSprop']:
        if opt_name == 'SGD':
            opt = optim.SGD
        else:
            opt = optim.RMSprop
        out_path, out_control, c_loss, d_loss, f_loss = trajopt(init_path, init_control, obs_list, lr, opt)
        # save the state and control
        file = open('trajopt_path_lr_%f_%s.pkl' % (lr, opt_name), 'wb')
        pickle.dump(out_path, file)
        file = open('trajopt_control_lr_%f_%s.pkl' % (lr, opt_name), 'wb')
        pickle.dump(out_control, file)
        file = open('trajopt_stat_lr_%f_%s.pkl' % (lr, opt_name), 'wb')
        pickle.dump([c_loss, d_loss, f_loss], file)
        file.close()

