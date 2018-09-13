import numpy as np
import cv2, os, sys
from copy import deepcopy
import pdb
import scipy.linalg as la



###############################################################################
############################################################################### Kalman Filter Functions/ Splines are loaded here too
def KalmanFilter(): # Generate Kalman Filter for the first time
    object_kalman1 = cv2.KalmanFilter(6,3,0)
    object_kalman1.measurementMatrix = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0]],np.float32)
    object_kalman1.transitionMatrix = np.array([[1,0,0,1,0,0],[0,1,0,0,1,0],[0,0,1,0,0,1],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]],np.float32)
    return(object_kalman1)



def KalmanChecking(OBEJCT3D, obejct_kalman, counter, flag_redo): # Updating Kalman Filter
    if counter == 0 or flag_redo == 1:
        for count in range(0,1000):
            obejct_kalman.correct(np.array([[np.float32(OBEJCT3D[0])],[np.float32(OBEJCT3D[1])],[np.float32(OBEJCT3D[2])]]))
            obejct_kalman_Predic = obejct_kalman.predict()

    obejct_kalman.correct(np.array([[np.float32(OBEJCT3D[0])],[np.float32(OBEJCT3D[1])],[np.float32(OBEJCT3D[2])]]))
    obejct_kalman_Predic = obejct_kalman.predict()
    return(obejct_kalman, obejct_kalman_Predic)



def Kalman_Spline(self): # Uploading Spline and Generating Kalman 
    for L0 in self.Coor_Labels[0::2]: setattr(getattr(self, L0), 'Kalman', KalmanFilter()) # Creating variables like self.EAR.Kalman/self.TAIL.Kalman/...

    running_code_Path = os.path.dirname(sys.argv[0])
    running_code_Path = os.path.join(running_code_Path, 'Required_files')
    self.xspline = np.load(os.path.join(running_code_Path, 'xspline_points.npy'), encoding="latin1").item()
    self.xspline_diff = deepcopy(self.xspline)
    for count in self.xspline_diff: self.xspline_diff[str(count)] = np.diff(self.xspline_diff[str(count)]) # Loading x splines

    self.yspline = np.load(os.path.join(running_code_Path, 'yspline_points.npy'), encoding="latin1").item()
    self.yspline_diff = deepcopy(self.yspline)
    for count in self.yspline_diff: self.yspline_diff[str(count)] = np.diff(self.yspline_diff[str(count)]) # loading y splines
    
    self.xspline_function = np.load(os.path.join(running_code_Path, 'xspline.npy'), encoding="latin1").item()
    self.yspline_function = np.load(os.path.join(running_code_Path, 'yspline.npy'), encoding="latin1").item()
    # self.spline_max_points = np.load(os.path.join(running_code_Path, 'spline_max.npy'), encoding="latin1").item()



###############################################################################
############################################################################### DLT functions
def DLT(self, Position, Coef, Cameras): # This function finds the 3D coordinates using DLT coeffs
    Cam1_Coe = Coef[..., int(Cameras[0])]; Cam1_Coe = np.append(Cam1_Coe,1)
    Cam2_Coe = Coef[...,int(Cameras[1])]; Cam2_Coe = np.append(Cam2_Coe,1);
    Coe = []
    Coe = [Cam1_Coe.reshape((3,4)),Cam2_Coe.reshape((3,4))]
    U12 = np.zeros([1,2]);V12 = np.zeros([1,2])
    U22 = np.zeros([1,2]);V22 = np.zeros([1,2])
    V12[0,0] = 700-Position[1]; V12[0,1] = 700-Position[3]
    if Cameras[0] == 0:
        U12[0,0] = Position[0]; U12[0,1] = Position[2]
    else:
        U12[0,0] = 2048-Position[0]; U12[0,1] = 2048-Position[2]

    m1 = np.zeros([4,3]); m2 = np.zeros([4,1])
    m1[0::2,0] = [U12[0,0]*Cam1_Coe[8]-Cam1_Coe[0],  U12[0,1]*Cam2_Coe[8]-Cam2_Coe[0]]
    m1[0::2,1] = [U12[0,0]*Cam1_Coe[9]-Cam1_Coe[1],  U12[0,1]*Cam2_Coe[9]-Cam2_Coe[1]]
    m1[0::2,2] = [U12[0,0]*Cam1_Coe[10]-Cam1_Coe[2], U12[0,1]*Cam2_Coe[10]-Cam2_Coe[2]]

    m1[1::2,0] = [V12[0,0]*Cam1_Coe[8]-Cam1_Coe[4],  V12[0,1]*Cam2_Coe[8]-Cam2_Coe[4]]
    m1[1::2,1] = [V12[0,0]*Cam1_Coe[9]-Cam1_Coe[5],  V12[0,1]*Cam2_Coe[9]-Cam2_Coe[5]]
    m1[1::2,2] = [V12[0,0]*Cam1_Coe[10]-Cam1_Coe[6], V12[0,1]*Cam2_Coe[10]-Cam2_Coe[6]]

    m2[0::2,0] = [Cam1_Coe[3]-U12[0,0], Cam2_Coe[3]-U12[0,1]]
    m2[1::2,0] = [Cam1_Coe[7]-V12[0,0], Cam2_Coe[7]-V12[0,1]]
    xyz = np.linalg.lstsq(m1,m2)[0]

    return(xyz)



def DLT_Inverse(self, xyz, Coef, Cameras): # This function finds the 2D projection on each camera using 3D coordinates and DLT coeffs
    Position = np.zeros([4])
    Cam1_Coe = Coef[...,int(Cameras[0])]; Cam1_Coe = np.append(Cam1_Coe,1); Cam2_Coe = Coef[...,int(Cameras[1])]; Cam2_Coe = np.append(Cam2_Coe,1);
    Coe = []; Coe = [Cam1_Coe.reshape((3,4)),Cam2_Coe.reshape((3,4))]

    temp = Coe[0].dot(np.append(xyz,1))
    if Cameras[0] == 0:
        Position[0] = float(temp[0]/temp[2])
    else:
        Position[0] = 2048-float(temp[0]/temp[2])
    Position[1] = 700-float(temp[1]/temp[2])

    temp = Coe[1].dot(np.append(xyz,1))
    if Cameras[0] == 0:
        Position[2] = float(temp[0]/temp[2])
    else:
        Position[2] = 2048-float(temp[0]/temp[2])
    Position[3] = 700-float(temp[1]/temp[2])

    return(Position)



def DLT_Based_Checking(self, OBJECT, OBJECT2, OBJECT3D, OBJECT_kalman, Cameras): # This function updates DLT and 3d Coordinates for landmarks
    Temp = DLT(self, np.append(np.array([float(OBJECT[1]),float(OBJECT[0])]), np.array([float(OBJECT2[1]),float(OBJECT2[0])])), self.Coef, Cameras)
    
    OBJECT3D[6:9,0] = OBJECT3D[0:3,0]
    OBJECT3D[0:3,0] = Temp[...,0]
    OBJECT3D[3:6,0] = OBJECT3D[0:3,0] - OBJECT3D[6:9,0]
    # 0:3 current frame 3D coordinates, 3:6 difference between last one the current frame
    # 6:9 previous frame 3D coordinates, 9:12 predicted 3D coordinates for the enxt frame
    if self.temp_counter == 1:
        flag_redo = 1
    else:
        flag_redo = 0
    (OBJECT_kalman, OBJECT_kalman_predict) = KalmanChecking(OBJECT3D, OBJECT_kalman, self.counter, flag_redo)
    OBJECT_kalman_Predict_Position = DLT_Inverse(self, OBJECT_kalman_predict[0:3,0], self.Coef, Cameras)
    OBJECT3D[9:12,0] = OBJECT_kalman_predict[0:3,0]

    return (OBJECT3D, OBJECT_kalman, OBJECT_kalman_predict, OBJECT_kalman_Predict_Position)





###############################################################################
###############################################################################
###############################################################################
############################################################################### LQR Class State
class State:
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v



def State_Update(state, a, delta, dt):
    if delta >= max_steer:
        delta = max_steer
    if delta <= - max_steer:
        delta = - max_steer
    state.x = state.x + state.v * math.cos(state.yaw) * dt
    state.y = state.y + state.v * math.sin(state.yaw) * dt
    state.yaw = state.yaw + state.v * math.tan(delta) * dt
    state.v = state.v + a * dt
    return state



###############################################################################
############################################################################### LQR Tracking
def PID_Control(target, current):
    Kp = 1
    a = Kp * (target - current)
    return a



def pi_2_pi(angle):
    while (angle > math.pi):
        angle = angle - 2.0 * math.pi

    while (angle < -math.pi):
        angle = angle + 2.0 * math.pi

    if angle > math.pi/2:
        angle = angle - math.pi

    if angle < -math.pi/2:
        angle = angle + math.pi
    return angle



def calc_speed_profile(cx, cy, cyaw, target_speed):
    speed_profile = [target_speed] * len(cx)

    direction = 1.0

    # Set stop point
    for i in range(len(cx) - 1):
        dyaw = cyaw[i + 1] - cyaw[i]
        switch = math.pi / 2.4 <= dyaw < math.pi / 2.0

        if switch:
            direction *= -1

        if direction != 1.0:
            speed_profile[i] = - target_speed
        else:
            speed_profile[i] = target_speed

        if switch:
            speed_profile[i] = 0.0

    speed_profile[-1] = 0.0

    #  flg, ax = plt.subplots(1)
    #  plt.plot(speed_profile, "-r")
    #  plt.show()

    return speed_profile



def Solve_DARE(A, B, Q, R):
    #solve a discrete time_Algebraic Riccati equation (DARE)
    X = Q
    # Number of Iteration
    maxiter = 30
    # Threshold for breaking the loop if the predicted position meeting the poisition of spline
    eps = 0.01
    for i in range(maxiter):
        Xn = A.T * X * A - A.T * X * B * \
                           la.inv(R + B.T * X * B) * B.T * X * A + Q
        if (abs(Xn - X)).max() < eps:
            X = Xn
            break
        X = Xn
    return Xn



def DQLR(A, B, Q, R):
    # x[k+1] = A x[k] + B u[k]; cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]; Trying to solve the ricatti equation
    X = Solve_DARE(A, B, Q, R)
    # compute the LQR gain
    K = np.matrix(la.inv(B.T * X * B + R) * (B.T * X * A))
    # find eig values and vectors
    eigVals, eigVecs = la.eig(A - B * K)
    return K, X, eigVals



def LQR_Control(state, cx, cy, cyaw, ck, pe, pth_e, dt):
    # calculate the trend direction
    ind, e = calc_nearest_index(state, cx, cy, cyaw) # NEEDS REPLACEMENT!!!!!!!!!!!!!!!!!!!!!!

    # can be calculated from the following equation for spline:
    # dx = self.sx.calcd(s), ddx = self.sx.calcdd(s), dy = self.sy.calcd(s), ddy = self.sy.calcdd(s), k = (ddy * dx - ddx * dy) / (dx ** 2 + dy ** 2)
    k = ck[ind] # NEEDS REPLACEMENT!!!!!!!!!!!!!!!!!!!!!!
    # v =  speed from the spline
    v = state.v # NEEDS REPLACEMENT!!!!!!!!!!!!!!!!!!!!!!
    # yaw = first dravative in y/ first drivative in x direction
    th_e = pi_2_pi(state.yaw - cyaw[ind])

    # unicycle_model.dt = 0.1--I think how small moving each time
    A = np.matrix(np.zeros((4, 4)))
    A[0, 0] = 1.0
    A[0, 1] = dt
    A[1, 2] = v
    A[2, 2] = 1.0
    A[2, 3] = dt

    # v = speed; unicycle_model.L = 0.5--length factor to make meter or something else
    B = np.matrix(np.zeros((4, 1)))
    B[3, 0] = v

    # Q and R are eye matrix with no changeing thing
    K, _, _ = DQLR(A, B, Q, R)

    x = np.matrix(np.zeros((4, 1)))

    # th-e is the previous yaw number and pe is the previous trend in the graph (positive or negative)
    x[0, 0] = e
    x[1, 0] = (e - pe)/dt
    x[2, 0] = th_e
    x[3, 0] = (th_e - pth_e)/dt
    
    # k is ck for the point
    ff = math.atan2(k, 1)
    fb = pi_2_pi((-K * x)[0, 0])

    delta = ff + fb

    return delta, ind, e, th_e



def calc_nearest_index(state, cx, cy, cyaw):
    dx = [state.x - icx for icx in cx]
    dy = [state.y - icy for icy in cy]

    d = [abs(math.sqrt(idx ** 2 + idy ** 2)) for (idx, idy) in zip(dx, dy)]

    mind = min(d)

    ind = d.index(mind)

    dxl = cx[ind] - state.x
    dyl = cy[ind] - state.y

    angle = pi_2_pi(cyaw[ind] - math.atan2(dyl, dxl))
    if angle < 0:
        mind *= -1

    return ind, mind



# This function should go inside the 
def closed_loop_prediction(cx, cy, cyaw, ck, speed_profile, goal):
    # Max simulation time sec
    dt = 0.1
    # Max time required
    T = 500.0  
    # Final point for tracking, reaching to destination
    goal_dis = 0.8
    # Increasing index for the next point on spline
    stop_speed = 0.05

    target_speed = 10

    # First time update for state
    state = State(x=-0.0, y=-0.0, yaw=0.0, v=0.0)

    # find the first point of spline (target_ind) -- it should come from my spline positioning
    time = 0.0; x = [state.x]; y = [state.y]; yaw = [state.yaw]; v = [state.v]; t = [0.0]; 
    target_ind = calc_nearest_index(state, cx, cy, cyaw)

    e, e_th  = 0.0, 0.0

    while T >= time:
        # finding dl which is the angle difference for yaw
        dl, target_ind, e, e_th = LQR_Control(state, cx, cy, cyaw, ck, e, e_th, dt)

        # findin the speed difference for adjustment 
        ai = PID_Control(speed_profile[target_ind], state.v)

        # updating the state using the new speed and angle difference
        state = State_Update(state, ai, dl, dt)

        if abs(state.v) <= stop_speed: target_ind += 1

        time = time + dt

        # check goal
        dx = state.x - goal[0]
        dy = state.y - goal[1]
        if math.sqrt(dx ** 2 + dy ** 2) <= goal_dis: break

        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)
        v.append(state.v)
        t.append(time)

    return t, x, y, yaw, v





    


