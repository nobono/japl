import numpy as np
import control as ct
import matplotlib.pyplot as plt
import scipy
from scipy import constants
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.base import OdeSolver
from autopilot import ss as apss
from ambiance import Atmosphere



A = np.array([
    [0, 0, 0,   1, 0, 0,    0, 0, 0,    0, 0, 0],       # xvel
    [0, 0, 0,   0, 1, 0,    0, 0, 0,    0, 0, 0],       # yvel
    [0, 0, 0,   0, 0, 1,    0, 0, 0,    0, 0, 0],       # zvel

    [0, 0, 0,   0, 0, 0,    apss.C[0][0], 0, 0,    apss.C[0][1], 0, 0], # xacc
    [0, 0, 0,   0, 0, 0,    0, apss.C[0][0], 0,    0, apss.C[0][1], 0], # yacc
    [0, 0, 0,   0, 0, 0,    0, 0, apss.C[0][0],    0, 0, apss.C[0][1]], # zacc

    [0, 0, 0,   0, 0, 0,    apss.A[0][0], 0, 0,    apss.A[0][1], 0, 0], # xacc_cmd
    [0, 0, 0,   0, 0, 0,    0, apss.A[0][0], 0,    0, apss.A[0][1], 0], # yacc_cmd
    [0, 0, 0,   0, 0, 0,    0, 0, apss.A[0][0],    0, 0, apss.A[0][1]], # zacc_cmd

    [0, 0, 0,   0, 0, 0,    apss.A[1][0], 0, 0,    apss.A[1][1], 0, 0], # xacc_cmd_dot
    [0, 0, 0,   0, 0, 0,    0, apss.A[1][0], 0,    0, apss.A[1][1], 0], # yacc_cmd_dot
    [0, 0, 0,   0, 0, 0,    0, 0, apss.A[1][0],    0, 0, apss.A[1][1]], # zacc_cmd_dot
    ])

# [ax, ay, az, ux, uy, uz]
B = np.array([
    [0, 0, 0,   0, 0, 0],
    [0, 0, 0,   0, 0, 0],
    [0, 0, 0,   0, 0, 0],
    [1, 0, 0,   0, 0, 0],
    [0, 1, 0,   0, 0, 0],
    [0, 0, 1,   0, 0, 0],
    [0, 0, 0,   *apss.B[0], 0, 0],
    [0, 0, 0,   0, *apss.B[0], 0],
    [0, 0, 0,   0, 0, *apss.B[0]],
    [0, 0, 0,   *apss.B[1], 0, 0],
    [0, 0, 0,   0, *apss.B[1], 0],
    [0, 0, 0,   0, 0, *apss.B[1]],
    ])

C = np.eye(12)

D = np.zeros((12, 6))

ss = ct.ss(A, B, C, D)


def unitize(vec):
    return vec / scipy.linalg.norm(vec)


def create_Rt(xx, yy, zz):
    mat = np.array([xx, yy, zz])
    return scipy.linalg.inv(mat)


def weave_maneuver(t, X):
    vm = unitize(X[3:6])
    base2 = [0, 0, 1]
    Rt = create_Rt(
            vm,
            np.cross(vm, base2),
            base2
            )
    ac = np.array([0, np.sin(0.5 * t), 0])
    return Rt @ ac


def popup_maneuver(t, X):
    vm = unitize(X[3:6])
    base2 = [1, 0, 0]
    Rt = create_Rt(
            vm,
            np.cross(vm, base2),
            base2
            )
    ac = np.zeros((3,))
    ac[1] = -1*np.sin(.3 * t)
    return Rt @ ac


def pronav(X, r_targ, v_targ, N=4.0):
    rm = X[:3]
    vm = X[3:6]
    v_r = v_targ - vm
    r = r_targ - rm
    omega = np.cross(r, v_r) / np.dot(r, r)
    ac = N * np.cross(v_r, omega)
    return ac


def guidance(t, X, r_targ):
    GLIMIT = 14.0
    rm = X[:3]
    ac = np.array([0, 0, 0])
    # ac = weave_maneuver(t, X)
    # ac_man = popup_maneuver(t, X)
    if 5e3 < rm[1] <= 10e3:
        if 0 < rm[2] <= 200:
            r_pop = np.array([0, 7e3, 200])
            ac = ac + pronav(X, r_pop, np.array([0, 0, 0]), N=3)
        if 7e3 < rm[1] <= 10e3:
            r_pop = np.array([0, 10e3, 10])
            ac = ac + pronav(X, r_pop, np.array([0, 0, 0]), N=6)
        # elif 10e3 < rm[1]:
        #     ac = ac + pronav(X, r_targ, np.array([0, 0, 0]))
    if scipy.linalg.norm(ac) > GLIMIT:
        ac = unitize(ac) * GLIMIT
    return ac


def dynamics(t, X, ss, r_targ):
    ac = guidance(t, X, r_targ)
    ac = [0, 5030, 0]
    CD = 0.4
    U = np.array([-(CD*X[3]**2), -(CD*X[4]**2), -(CD*X[5]**2),
                  *ac])
    Xdot = ss.A @ X + ss.B @ U
    return Xdot


# Inits
####################################
t_span = [0, 300]
x0 = np.zeros((12))
x0[:3] = np.array([0, 0, 10])    #R0
x0[3:6] = np.array([0, 200, 0])   #V0

targ_R0 = np.array([0, 20e3, 0])
####################################


# Events
####################################
def hit_ground_event(t, X, ss, r_targ):
    return X[2]
hit_ground_event.terminal = True

def hit_target_event(t, X, ss, r_targ):
    rm = X[:3]
    hit_dist = r_targ - rm
    return  hit_dist[0] + hit_dist[1] + hit_dist[2]
    
hit_target_event.terminal = True
####################################


sol = solve_ivp(
        dynamics,
        t_span=t_span,
        t_eval=np.linspace(t_span[0], t_span[1], 10000),
        y0=x0,
        args=(ss, targ_R0),
        events=[
            hit_target_event,
            hit_ground_event,
            ]
        )

t = sol['t']
y = sol['y'].T

# state = State(y, t)
# velmag = [scipy.linalg.norm(i) for i in y[:, 2:4]]


# fig, (ax, ax2, ax3) = plt.subplots(3, figsize=(10, 8))
# ax.plot(y[:, 1], y[:, 2])
# ax.set_title("xy")

# ax2.plot(y[:, 1], y[:, 5])
# ax2.set_title("yvel")

# ax3.plot(t, y[:, 5])
# ax3.set_title("ac")

r_pop1 = np.array([0, 7e3, 300])
r_pop2 = np.array([0, 10e3, 10])

# 3D Plot
# fig, (ax, ax2)= plt.subplots(2, figsize=(10, 8))
fig = plt.figure(figsize=(15, 10))
ax = plt.axes(projection='3d')
# ax = fig.add_subplot(2, 1, 1, projection='3d')
ax.plot3D(y[:, 0], y[:, 1], y[:, 2])
ax.plot3D(*targ_R0, marker='.')
ax.plot3D(*r_pop1, marker='.', color='green')
ax.plot3D(*r_pop2, marker='.', color='red')
ax.set_xlabel("E")
ax.set_ylabel("N")
ax.set_zlabel("D")

fig2, (ax2, ax3, ax4) = plt.subplots(3, figsize=(10, 8))
# ax2.plot(y[:, 0], y[:, 1])
# ax2.set_title("xy")

ax3.plot(y[:, 1], y[:, 4])
ax3.set_title("yvel")
ax4.plot(y[:, 1], y[:, 5])
ax4.set_title("zvel")

plt.show()

