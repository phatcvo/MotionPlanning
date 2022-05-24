# Program by Phat C. Vo
import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from re import T

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../MotionPlanning/")

import Control.draw as draw
import CurvesGenerator.reeds_shepp as rs
import CurvesGenerator.cubic_spline as cs


class ini:
    # PID config
    Kp = 0.3
    Ki = 0.001
    Kd = 0.00001
    int_term = 0
    derivative_term = 0
    last_error = None

    # System config
    Ld = 1.6  # look ahead distance
    kf = 0.4  # look forward gain
    dt = 0.1  # T step
    dist_stop = 0.2  # stop distance

    MAX_STEER = 0.30
    MAX_ACCELERATION = 5.0

    # Vehicle config
    RF = 1.2  # [m] distance from rear to vehicle front end of vehicle
    RB = 0.1  # [m] distance from rear to vehicle back end of vehicle
    W = 1  # [m] width of vehicle
    WD = 0.95 * W  # [m] distance between left-right wheels
    WB = 1 #2.5  # [m] Wheel base
    TR = 0.27  # [m] Tyre radius
    TW = 0.54  # [m] Tyre width


class Node: 
    def __init__(self, x, y, yaw, v, direct):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.direct = direct

    @staticmethod
    def limit_input(delta):
        if delta > 1.2 * C.MAX_STEER:
            return 1.2 * C.MAX_STEER

        if delta < -1.2 * C.MAX_STEER:
            return -1.2 * C.MAX_STEER

        return delta

    def update(self, acc, delta, direct):
        delta = self.limit_input(delta)
        self.x += self.v * math.cos(self.yaw) * ini.dt
        self.y += self.v * math.sin(self.yaw) * ini.dt
        self.yaw += self.v / ini.WB * math.tan(delta) * ini.dt
        self.direct = direct
        self.v += self.direct * a * ini.dt
    
class Nodes:
    def __init__(self):
        self.x = []
        self.y = []
        self.yaw = []
        self.v = []
        self.t = []
        self.direct = []

    def add(self, t, node):
        self.x.append(node.x)
        self.y.append(node.y)
        self.yaw.append(node.yaw)
        self.v.append(node.v)
        self.t.append(t)
        self.direct.append(node.direct)

class PATH:
    def __init__(self, cx, cy, cyaw):
        self.cx = cx
        self.cy = cy
        self.cyaw = cyaw
        self.ind_end = len(self.cx) - 1
        self.ind_old = None
    
    def calc_distance (self, node, ind):
        dis = math.hypot(node.x - self.cx[ind], node.y - self.cy[ind])
        return dis

    # calc index of the nearest point to current position
    def calc_nearest_ind(self, node):
        dx = [node.x - x for x in self.cx]
        dy = [node.y - y for y in self.cy]
        ind = np.argmin(np.hypot(dx, dy))
        self.ind_old = ind

    # search index of target point in the reference path.
    def target_index(self, node):
        if self.ind_old is Node:
            self.calc_nearest_ind(node)
        
        Lf = ini.kf * node.v + ini.Ld

        for ind in range(self.ind_old, self.ind_end + 1):
            if self.calc_distance(node, ind) > Lf:
                self.ind_old = ind
                return ind, Lf
        self.ind_old = self.ind_end

        return self.ind_end, Lf

# Controller ======================================

def pure_pursuit(node, ref_path, ind_old):
    # target point and pursuit distance
    ind, Lf = ref_path.target_index(node)
    ind = max(ind, ind_old)

    tx = ref_path.cx[ind]
    ty = ref_path.cy[ind]
    
    alpha = math.atan2(ty - node.y, tx - node.x) - node.yaw
    delta = math.atan2(2.0 * C.WB * math.sin(alpha), Lf)

    return delta, ind


def pi_2_pi(angle):
    if angle > math.pi:
        return angle - 2.0 * math.pi
    if angle < -math.pi:
        return angle + 2.0 * math.pi

    return angle


# PID controller and design speed profile.  
def pid_control(target_v, v, dist, direct):
    error = (target_v - direct * v)
    C.int_term += error*C.Ki*C.dt
    if C.last_error is not None:
        C.derivative_term = (error-C.last_error)/C.dt*C.Kd
    C.last_error = error

    a = C.Kp * error + C.int_term + C.derivative_term

    if dist < 10.0:
        if v > 3.0:
            a = -2.5
        elif v < -2.0:
            a = -1.0

    return a # desired acceleration


def generate_path(s):
    
    path_x, path_y, yaw, direct, rc = [], [], [], [], []
    x_rec, y_rec, yaw_rec, direct_rec, rc_rec, vel_rec, steer_angle_rec = [], [], [], [], [], [], []
    direct_flag = 1.0
    max_c = math.tan(C.MAX_STEER) / C.WB  # max curvature

    for i in range(len(s) - 1):
        s_x, s_y, s_yaw = s[i][0], s[i][1], np.deg2rad(s[i][2])
        g_x, g_y, g_yaw = s[i + 1][0], s[i + 1][1], np.deg2rad(s[i + 1][2])
        path_i = rs.calc_optimal_path(s_x, s_y, s_yaw, g_x, g_y, g_yaw, max_c)
        irc, rds = rs.calc_curvature(path_i.x, path_i.y, path_i.yaw, path_i.directions)

        ix = path_i.x
        iy = path_i.y
        iyaw = path_i.yaw
        idirect = path_i.directions

        for j in range(len(ix)):
            if idirect[j] == direct_flag:
                x_rec.append(ix[j])
                y_rec.append(iy[j])
                yaw_rec.append(iyaw[j])
                direct_rec.append(idirect[j])
                rc_rec.append(irc[j])
            else:
                if len(x_rec) == 0 or direct_rec[0] != direct_flag:
                    direct_flag = idirect[j]
                    continue

                path_x.append(x_rec)
                path_y.append(y_rec)
                yaw.append(yaw_rec)
                direct.append(direct_rec)
                rc.append(rc_rec)
                x_rec, y_rec, yaw_rec, direct_rec = [x_rec[-1]], [y_rec[-1]], [yaw_rec[-1]], [-direct_rec[-1]]

    path_x.append(x_rec)
    path_y.append(y_rec)
    yaw.append(yaw_rec)
    direct.append(direct_rec)
    rc.append(rc_rec)

    x_all, y_all = [], []

    for ix, iy in zip(path_x, path_y):
        x_all += ix
        y_all += iy

    return path_x, path_y, yaw, direct, rc, x_all, y_all

def main():
    # generate path: [x, y, yaw]
    states = [(0, 5, 0), (20, 15, 0), (35, 20, 90), (40, 0, 180),
              (20, -10, 180), (0, -20, 270), (-5, -5, 0), (20, 5, 45)]
    x_ref, y_ref, yaw_ref, direct, curv, x_all, y_all = generate_path(states)

    # simulation
    maxTime = 100.0
    yaw_old = 0.0
    x0, y0, yaw0, direct0 = 0, 0, yaw_ref[0][0], direct[0][0]
    x_rec, y_rec, vel_rec, steer_angle_rec, delta_rec, delta_rec1, delta_rec2 = [], [], [], [], [], [], []

    for cx, cy, cyaw, cdirect, ccurv in zip(x_ref, y_ref, yaw_ref, direct, curv):
        t = 0.0
        node = Node(x=x0, y=y0, yaw=yaw0, v=0.0, direct=direct0)
        nodes = Nodes()
        nodes.add(t, node)
        ref_trajectory = PATH(cx, cy, cyaw, ccurv)
        target_ind, _ = ref_trajectory.target_index(node)

        while t <= maxTime:
            if cdirect[0] > 0:
                target_speed = 30.0 / 3.6
                C.Ld = 4.0
                C.dist_stop = 1.5
                C.dc = -1.1
            else:
                target_speed = 20.0 / 3.6
                C.Ld = 2.5
                C.dist_stop = 0.2
                C.dc = 0.2

            xt = node.x + C.dc * math.cos(node.yaw)
            yt = node.y + C.dc * math.sin(node.yaw)
            dist = math.hypot(xt - cx[-1], yt - cy[-1])

            if dist < C.dist_stop:
                break

            acceleration = pid_control(target_speed, node.v, dist, cdirect[0])
            delta1, target_index = pure_pursuit(node, ref_trajectory, target_ind)

            # delta2, ind = rear_wheel_feedback_control(node, ref_trajectory)
            delta3, target_index = front_wheel_feedback_control(node, ref_trajectory)
            print("delta3:", cyaw)
            # delta = 1.0*delta1 + 0*delta2

            # delta_rec.append(delta)
            delta_rec1.append(delta1)
            delta_rec2.append(delta3)

            t += C.dt
            node.update(acceleration, delta1, cdirect[0])
            nodes.add(t, node)

            
            x_rec.append(node.x)
            y_rec.append(node.y)

            dy = (node.yaw - yaw_old) / (node.v * C.dt)
            steer = rs.pi_2_pi(-math.atan(C.WB * dy))
            vel_rec.append(node.v)
            steer_angle_rec.append(steer*180/math.pi)

            yaw_old = node.yaw

            x0 = nodes.x[-1]
            y0 = nodes.y[-1]
            yaw0 = nodes.yaw[-1]
            direct0 = nodes.direct[-1]

            # animation
            plt.cla()
            plt.plot(x_all, y_all, color='gray', linewidth=2, label = 'path')
            plt.plot(x_rec, y_rec, color='blue', linewidth=1, label = 'track')
            plt.plot(cx[target_index], cy[target_index], ".r")
            # plt.plot(cx[ind], cy[ind], '.c')
            draw.draw_car(node.x, node.y, yaw_old, steer, C)

            for m in range(len(states)):
                draw.Arrow(states[m][0], states[m][1], np.deg2rad(states[m][2]), 2, 'darkviolet')

            plt.axis("equal")
            plt.title("v =" + str(node.v * 3.6)[:4] + "km/h")
            plt.gcf().canvas.mpl_connect('key_release_event',
                                         lambda event:
                                         [exit(0) if event.key == 'escape' else None])
            plt.legend()
            plt.pause(0.001)

    fig1, (ax1, ax2, ax3) = plt.subplots(3)
    ax1.plot(np.arange(0, len(vel_rec)), vel_rec,'--b', label ='Velocity')
    ax2.plot(np.arange(0, len(steer_angle_rec)), steer_angle_rec, c ='r', label ='Steer angle')
    ax3.plot(np.arange(0, len(delta_rec)), delta_rec,'--k',label ='Adap')
    ax3.plot(np.arange(0, len(delta_rec1)), delta_rec1,'-b', label ='PP')
    ax3.plot(np.arange(0, len(delta_rec1)), delta_rec2,'--r',label ='ST')
    ax1.legend()
    ax2.legend()
    ax3.legend()
            
    plt.show()
