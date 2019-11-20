import torch
import numpy as np
from tools.utility import *
import time
DEFAULT_STEP = 2.
import matplotlib.pyplot as plt

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_autoscale_on(True)

hl, = ax.plot([], [])
hl_sample, = ax.plot([], [], color='g')
def update_line(h, ax, new_data):
    h.set_xdata(np.append(h.get_xdata(), new_data[0]))
    h.set_ydata(np.append(h.get_ydata(), new_data[1]))
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()
    #plt.draw()
def update_lines(hs, ax, new_datas):
    for i in range(len(hs)):
        new_data = new_datas[i]
        h = hs[i]
        h.set_xdata(np.append(h.get_xdata(), new_data[0]))
        h.set_ydata(np.append(h.get_ydata(), new_data[1]))
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()

def removeCollision(path, obc, IsInCollision):
    new_path = []
    # rule out nodes that are already in collision
    for i in range(0,len(path)):
        if not IsInCollision(path[i],obc):
            new_path.append(path[i])
    return new_path

def steerTo(bvp_solver, start, end, obc, IsInCollision, step_sz=DEFAULT_STEP):
    # test if there is a collision free path from start to end, with step size
    # given by step_sz, and with generic collision check function
    # here we assume start and end are tensors
    # return 0 if in coliision; 1 otherwise
    res = bvp_solver.steerTo(start, end, 100, 20, 200, 0.002)
    # check if endpoint is the same as end
    # for pendulum the first dimension is circurtry
    dist = res[0][-1] - end
    if dist[0] > np.pi:
        dist[0] = 2*np.pi - dist[0]
    dist = np.linalg.norm(dist)
    #dist = np.linalg.norm(res[0][-1] - end)
    print('steerTo distance: %f' % (dist))
    if dist > 0.1:
        steer = False
    else:
        steer = True
    state = []
    control = []
    time_step = []
    for i in range(len(res[0])):
        state.append(res[0][i])
    for i in range(len(res[1])):
        control.append(res[1][i])
    for i in range(len(res[2])):
        time_step.append(res[2][i])
    return steer, state, control, time_step

def feasibility_check(bvp_solver, path, obc, IsInCollision, step_sz=DEFAULT_STEP):
    # checks the feasibility of entire path including the path edges
    # by checking for each adjacent vertices
    for i in range(0,len(path)-1):
        steer, _, _, _ = steerTo(bvp_solver, path[i],path[i+1],obc,IsInCollision,step_sz=step_sz)
        if not steer:
            # collision occurs from adjacent vertices
            return 0
    return 1

def lvc(path, obc, IsInCollision, step_sz=DEFAULT_STEP):
    # lazy vertex contraction
    for i in range(0,len(path)-1):
        for j in range(len(path)-1,i+1,-1):
            ind=0
            ind=steerTo(bvp_solver,path[i],path[j],obc,IsInCollision,step_sz=step_sz)
            if ind==1:
                pc=[]
                for k in range(0,i+1):
                    pc.append(path[k])
                for k in range(j,len(path)):
                    pc.append(path[k])
                return lvc(pc,obc,IsInCollision,step_sz=step_sz)
    return path

def neural_replan(mpNet, bvp_solver, path, control, time_step, obc, obs, IsInCollision, normalize, unnormalize, init_plan_flag, step_sz=DEFAULT_STEP, time_flag=False):
    """
        Modify the original MPNet planning to simple line search algorithm.
        No need to varify if consecutive points are connected, because they are
        ensured to be connected in neural_replanner
    """
    # for plotting
    global hl, hl_sample, ax, fig, hl_g
    ax.clear()
    ax.plot(path[0][0], path[0][1], 'rx')
    ax.plot(path[-1][0], path[-1][1], 'bx') # goal
    hl, = ax.plot([], [])
    hl_sample, = ax.plot([], [], color='r')

    if init_plan_flag:
        # if it is the initial plan, then we just do neural_replan
        MAX_LENGTH = 80
        res, mini_path, mini_control, mini_time, time_d = neural_replanner_line(mpNet, bvp_solver, path[0], path[-1], obc, obs, IsInCollision, \
                                            normalize, unnormalize, MAX_LENGTH, step_sz=step_sz)
        if res:
            if time_flag:
                return res, removeCollision(mini_path, obc, IsInCollision), mini_control, mini_time, time_d
            else:
                return res, removeCollision(mini_path, obc, IsInCollision), mini_control, mini_time
        else:
            # can't find a path
            if time_flag:
                return res, path, [], [], time_d
            else:
                return res, path, [], []
    MAX_LENGTH = 50
    # replan segments of paths
    new_path = [path[0]]
    new_control = control
    new_time_step = time_step
    time_norm = 0.
    """
    for i in range(len(path)-1):
        # look at if adjacent nodes can be connected
        # assume start is already in new path
        start = path[i]
        goal = path[i+1]
        steer = steerTo(start, goal, obc, IsInCollision, step_sz=step_sz)
        if steer:
            new_path.append(goal)
        else:
            # plan mini path
            mini_path, time_d = neural_replanner(mpNet, start, goal, obc, obs, IsInCollision, \
                                                normalize, unnormalize, MAX_LENGTH, step_sz=step_sz)
            time_norm += time_d
            if mini_path:
                new_path += removeCollision(mini_path[1:], obc, IsInCollision)  # take out start point
            else:
                new_path += path[i+1:]     # just take in the rest of the path
                break
    """
    res, mini_path, mini_control, mini_time, time_d = neural_replanner_line(mpNet, bvp_solver, path[-2], path[-1], obc, obs, IsInCollision, \
                                        normalize, unnormalize, MAX_LENGTH, step_sz=step_sz)
    time_norm += time_d
    if res:
        #new_path += removeCollision(mini_path[1:], obc, IsInCollision)
        new_path += mini_path[1:]
        new_control += mini_control
        new_time_step += steer_time
    else:
        new_path.append(path[-1])

    if time_flag:
        return res, new_path, new_control, new_time_step, time_norm
    else:
        return res, new_path, new_control, new_time_step

def neural_replanner_line(mpNet, bvp_solver, start, goal, obc, obs, IsInCollision, normalize, unnormalize, MAX_LENGTH, step_sz=DEFAULT_STEP):
    # plan a mini path from start to goal
    # obs: tensor
    """
        simple line search algorithm for planning
    """
    global hl, hl_sample
    itr=0
    pA=[]
    pA.append(start)
    pB=[]
    pB.append(goal)
    target_reached=0
    tree=0
    new_path = []
    time_norm = 0.
    control = []
    time_step = []
    print('outside:')
    print('start:')
    print(start)
    while target_reached==0 and itr<MAX_LENGTH:
        print('iteration: %d' % (itr))
        print('goal:')
        print(goal)
        print('start state:')
        print(start)
        # update the path
        update_line(hl, ax, start)


        itr=itr+1  # prevent the path from being too long
        ip1 = np.concatenate([start, goal])
        np.expand_dims(ip1, 0)
        #ip1=torch.cat((obs,start,goal)).unsqueeze(0)
        time0 = time.time()
        ip1=normalize(ip1)
        ip1 = torch.FloatTensor(ip1)
        time_norm += time.time() - time0
        ip1=to_var(ip1)
        if obs is not None:
            obs = torch.FloatTensor(obs).unsqueeze(0)
            obs=to_var(obs)
        start_sample=mpNet(ip1,obs).squeeze(0)
        # unnormalize to world size
        start_sample=start_sample.data.cpu().numpy()
        time0 = time.time()
        start_sample = unnormalize(start_sample)
        time_norm += time.time() - time0


        if itr % 10 == 0:
            # use endpoint
            start_sample = goal
        print('start sample:')
        print(start_sample)
        #update_lines([hl, hl_sample], ax, [start, start_sample])
        update_line(hl_sample, ax, start_sample)

        # connect to start through trajopt
        steer, steer_state, steer_control, steer_time_step = steerTo(bvp_solver, start, start_sample, obc, IsInCollision, step_sz=step_sz)
        #print(steer_state)
        print('steer_control:')
        print(steer_control)
        print('steer_time_step:')
        print(steer_time_step)
        #for i in range(1,len(steer_state)):
        #    update_line(hl, ax, steer_state[i])
        pA += steer_state[1:]
        control += steer_control
        time_step += steer_time_step
        start = steer_state[-1]
        target_reached, to_goal_state, to_goal_control, to_goal_time_step =steerTo(bvp_solver, start, goal, obc, IsInCollision, step_sz=step_sz)
    #if target_reached==0:
    #    return 0, time_norm
    #else:
    for p1 in range(len(pA)):
        new_path.append(pA[p1])
    for p2 in range(len(pB)-1,-1,-1):
        new_path.append(pB[p2])

    return target_reached, new_path, control, time_step, time_norm


def neural_replanner(mpNet, start, goal, obc, obs, IsInCollision, normalize, unnormalize, MAX_LENGTH, step_sz=DEFAULT_STEP):
    # plan a mini path from start to goal
    # obs: tensor
    """

    """
    itr=0
    pA=[]
    pA.append(start)
    pB=[]
    pB.append(goal)
    target_reached=0
    tree=0
    new_path = []
    time_norm = 0.
    while target_reached==0 and itr<MAX_LENGTH:
        itr=itr+1  # prevent the path from being too long
        if tree==0:
            ip1 = torch.cat((start, goal)).unsqueeze(0)
            ob1 = torch.FloatTensor(obs).unsqueeze(0)
            #ip1=torch.cat((obs,start,goal)).unsqueeze(0)
            time0 = time.time()
            ip1=normalize(ip1)
            time_norm += time.time() - time0
            ip1=to_var(ip1)
            ob1=to_var(ob1)
            start=mpNet(ip1,ob1).squeeze(0)
            # unnormalize to world size
            start=start.data.cpu()
            time0 = time.time()
            start = unnormalize(start)
            time_norm += time.time() - time0
            pA.append(start)
            tree=1
        else:
            ip2 = torch.cat((goal, start)).unsqueeze(0)
            ob2 = torch.FloatTensor(obs).unsqueeze(0)
            #ip2=torch.cat((obs,goal,start)).unsqueeze(0)
            time0 = time.time()
            ip2=normalize(ip2)
            time_norm += time.time() - time0
            ip2=to_var(ip2)
            ob2=to_var(ob2)
            goal=mpNet(ip2,ob2).squeeze(0)
            # unnormalize to world size
            goal=goal.data.cpu()
            time0 = time.time()
            goal = unnormalize(goal)
            time_norm += time.time() - time0
            pB.append(goal)
            tree=0
        target_reached=steerTo(start, goal, obc, IsInCollision, step_sz=step_sz)

    if target_reached==0:
        return 0, time_norm
    else:
        for p1 in range(len(pA)):
            new_path.append(pA[p1])
        for p2 in range(len(pB)-1,-1,-1):
            new_path.append(pB[p2])

    return new_path, time_norm


def complete_replan_global(mpNet, path, true_path, true_path_length, obc, obs, obs_i, \
                           normalize, step_sz=DEFAULT_STEP):
    # use the training dataset as demonstration (which was trained by rrt*)
    # input path: list of tensor
    # obs: tensor
    demo_path = true_path[:true_path_length]
    dataset, targets, env_indices = transformToTrain(demo_path, len(demo_path), obs, obs_i)
    added_data = list(zip(dataset,targets,env_indices))
    bi = np.array(dataset).astype(np.float32)
    bobs = obs.numpy().reshape(1,-1).repeat(len(dataset),axis=0).astype(np.float32)
    bi = torch.FloatTensor(bi)
    bobs = torch.FloatTensor(bobs)
    bt = torch.FloatTensor(targets)
    # normalize first
    bi, bt = normalize(bi), normalize(bt)
    mpNet.zero_grad()
    bi=to_var(bi)
    bobs=to_var(bobs)
    bt=to_var(bt)
    mpNet.observe(0, bi, bobs, bt)
    demo_path = [torch.from_numpy(p).type(torch.FloatTensor) for p in demo_path]
    return demo_path, added_data


def transformToTrain(path, path_length, obs, obs_i):
    dataset=[]
    targets=[]
    env_indices = []
    for m in range(0, path_length-1):
        data = np.concatenate( (path[m], path[path_length-1]) ).astype(np.float32)
        targets.append(path[m+1])
        dataset.append(data)
        env_indices.append(obs_i)
    return dataset,targets,env_indices
