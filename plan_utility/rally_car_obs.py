import torch
import numpy as np
import sys
sys.path.append('..')
from plan_utility.line_line_cc import line_line_cc

def normalize(x, bound):
    # normalize to -1 ~ 1  (bound can be a tensor)
    #return x
    bound = torch.tensor(bound)
    if len(x.size()) > 1:
        if len(x[0]) != len(bound):
            x[:,:-len(bound)] = x[:,:-len(bound)] / bound
            x[:,-len(bound):] = x[:,-len(bound):] / bound
        else:
            x = x / bound
    else:
        if len(x) != len(bound):
            x[:-len(bound)] = x[:-len(bound)] / bound
            x[-len(bound):] = x[-len(bound):] / bound
        else:
            x = x / bound
    return x
def unnormalize(x, bound):
    # normalize to -1 ~ 1  (bound can be a tensor)
    # x only one dim
    #return x
    bound = torch.tensor(bound)
    if len(x.size()) > 1:
        if len(x[0]) != len(bound):
            x[:,:-len(bound)] = x[:,:-len(bound)] * bound
            x[:,-len(bound):] = x[:,-len(bound):] * bound
        else:
            x = x * bound
    else:
        if len(x) != len(bound):
            x[:-len(bound)] = x[:-len(bound)] * bound
            x[-len(bound):] = x[-len(bound):] * bound
        else:
            x = x * bound
    return x


def dynamics(state, control):
    '''
    implement the function x_dot = f(x,u)
    return the derivative w.r.t. x
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
    
    _vx = state[2]
    _vy = state[3]
    _theta = state[4]
    _thetadot = state[5]
    _wf = state[6]
    _wr = state[7]

    _sta = control[CONTROL_STA]
    _tf = control[CONTROL_TF]
    _tr = control[CONTROL_TR]

    deriv = np.zeros(8)

    deriv[STATE_X] = _vx
    deriv[STATE_Y] = _vy
    deriv[STATE_THETA] = _thetadot

    V = np.sqrt(_vx*_vx+_vy*_vy)
    beta = np.arctan2(_vy,_vx) - _theta
    V_Fx = V*np.cos(beta-_sta) + _thetadot*LF*np.sin(_sta)
    V_Fy = V*np.sin(beta-_sta) + _thetadot*LF*np.cos(_sta)
    V_Rx = V*np.cos(beta)
    V_Ry = V*np.sin(beta) - _thetadot*LR

    s_Fx = (V_Fx - _wf*R)/(_wf*R)
    s_Fy = V_Fy/(_wf*R)
    s_Rx = (V_Rx - _wr*R)/(_wr*R)
    s_Ry = V_Ry/(_wr*R)

    s_F = np.sqrt(s_Fx*s_Fx+s_Fy*s_Fy)
    s_R = np.sqrt(s_Rx*s_Rx+s_Ry*s_Ry)

    mu_F = D*np.sin(C*atan(B*s_F))
    mu_R = D*np.sin(C*atan(B*s_R))
    if np.isfinite(s_Fx):
        mu_Fx = -1*(s_Fx/s_F)*mu_F
    else:
        mu_Fx = -mu_F
    if np.isfinite(s_Fy):
        mu_Fy = -1*(s_Fy/s_F)*mu_F
    else:
        mu_Fy = -mu_F
    if np.isfinite(s_Rx):
        mu_Rx = -1*(s_Rx/s_R)*mu_R
    else
        mu_Rx = -mu_R
    if np.isfinite(s_Ry):
        mu_Ry = -1*(s_Ry/s_R)*mu_R
    else:
        mu_Ry = -mu_R
    fFz = (LR*M*(9.8) - H*M*9.8*mu_Rx) / (LF+LR+H*(mu_Fx*np.cos(_sta)-mu_Fy*np.sin(_sta)-mu_Rx))
    fRz = M*9.8 - fFz

    fFx = mu_Fx * fFz
    fFy = mu_Fy * fFz
    fRx = mu_Rx * fRz
    fRy = mu_Ry * fRz

    deriv[STATE_VX] = (fFx*np.cos(_theta+_sta)-fFy*np.sin(_theta+_sta)+fRx*np.cos(_theta)-fRy*np.sin(_theta) )/M
    deriv[STATE_VY] = (fFx*np.sin(_theta+_sta)+fFy*np.cos(_theta+_sta)+fRx*np.sin(_theta)+fRy*np.cos(_theta) )/M
    deriv[STATE_THETADOT] = ((fFy*np.cos(_sta)+fFx*np.sin(_sta))*LF - fRy*LR)/IZ
    deriv[STATE_WF] = (_tf-fFx*R)/IF
    deriv[STATE_WR] = (_tr-fRx*R)/IR
    return deriv

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
    if x[0] < MIN_X or x[0] > MAX_X or x[1] < MIN_Y or x[1] > MAX_Y:
        return True
        
    robot_corner=np.zeros((4,2),dtype=np.float32)
    robot_axis=np.zeros((2,2),dtype=np.float32)
    robot_orign=np.zeros(2,dtype=np.float32)
    length=np.zeros(2,dtype=np.float32)
    X1=np.zeros(2,dtype=np.float32)
    Y1=np.zeros(2,dtype=np.float32)

    X1[0]=math.cos(x[STATE_THETA])*(WIDTH/2.0)
    X1[1]=-math.sin(x[STATE_THETA])*(WIDTH/2.0)
    Y1[0]=math.sin(x[STATE_THETA])*(LENGTH/2.0)
    Y1[1]=math.cos(x[STATE_THETA])*(LENGTH/2.0)

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
        X=np.zeros(2,dtype=np.float32)
        Y=np.zeros(2,dtype=np.float32)
        length2=np.zeros(2,dtype=np.float32)

        X[0]=1.0*size/2.0
        X[1]=0.0
        Y[0]=0.0
        Y[1]=1.0*size/2.0

        for j in range(0,2):
            #obs_corner[0][j]=obc[i][j]-X[j]-Y[j]
            #obs_corner[1][j]=obc[i][j]+X[j]-Y[j]
            #obs_corner[2][j]=obc[i][j]+X[j]+Y[j]
            #obs_corner[3][j]=obc[i][j]-X[j]+Y[j]
            obs_corner[0][j] = obc[j]
            obs_corner[1][j] = obc[2+j]
            obs_corner[2][j] = obc[2*2+j]
            obs_corner[3][j] = obc[3*2+j]
            

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



