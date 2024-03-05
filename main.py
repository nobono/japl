import numpy as np
import control as ct
import matplotlib.pyplot as plt
import scipy
from scipy import constants
from scipy.linalg import norm
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
    vec = np.asarray(vec)
    return vec / norm(vec)


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


gd_phase = 0

def popup_maneuver(t, X, r_targ, ac):
    # vm = unitize(X[3:6])
    # base2 = [1, 0, 0]
    # Rt = create_Rt(
    #         vm,
    #         np.cross(vm, base2),
    #         base2
    #         )
    # ac = np.zeros((3,))
    # ac[1] = -1*np.sin(.3 * t)
    # return Rt @ ac
    START_POP_RANGE = 6.5e3
    STOP_POP_ALT = 90
    START_DIVE_RANGE = 8e3
    STOP_DIVE_ALT = 30 # 60
    rm = X[:3]
    vm = X[3:6]
    global gd_phase
    match gd_phase:
        case 0 :
            r_pop = np.array([0, START_POP_RANGE, 90])
            ac = ac + pronav(X, r_pop, np.array([0, 0, 0]), N=4)
            if rm[2] >= STOP_POP_ALT:
                gd_phase += 1
        case 1 :
            r_pop = np.array([0, START_DIVE_RANGE, 10])
            ac = ac + pronav(X, r_pop, np.array([0, 0, 0]), N=3)
            if rm[2] <= STOP_DIVE_ALT:
                gd_phase += 1
        case 2 :
            # Kp = 80.0
            # Kp_rz = 0.0006
            # rz_err = Kp_rz * (max(min(10 - rm[2], 10), -10) / 10)
            # vmd_hat = unitize([0, 1, rz_err])
            # vm_hat = unitize(vm)
            # vm_err = vmd_hat - vm_hat
            # ac = ac + Kp * vm_err
            r_pop = np.array([0, 13e3, 10])
            ac = ac + pronav(X, r_pop, np.array([0, 0, 0]), N=40)
            if rm[1] > 12e3:
                gd_phase += 1
        case 3 :
            ac = ac + pronav(X, r_targ, np.array([0, 0, 0]), N=4)
        case _ :
            pass
    return ac


def pronav(X, r_targ, v_targ, N=4.0):
    rm = X[:3]
    vm = X[3:6]
    v_r = v_targ - vm
    r = r_targ - rm
    omega = np.cross(r, v_r) / np.dot(r, r)
    ac = N * np.cross(v_r, omega)
    return ac


def atmosphere_model(t, X):
    rm = X[:3]
    vm = X[3:6]
    ATMOS_BOUNDS = [-5e3, 81e3]
    atmos = Atmosphere(min(max((rm[2] / 1000.0), ATMOS_BOUNDS[0]), ATMOS_BOUNDS[1]))
    CD = 0.45
    A = .25**2
    xfd = -(0.5 * CD * A * atmos.density[0] * vm[0])
    yfd = -(0.5 * CD * A * atmos.density[0] * vm[1])
    zfd = -(0.5 * CD * A * atmos.density[0] * vm[2])
    return np.array([xfd, yfd, zfd])


def guidance(t, X, r_targ):
    GLIMIT = 14.0
    rm = X[:3]
    vm = X[3:6]
    ac = unitize(vm) * 3.5
    # ac = weave_maneuver(t, X)
    # ac_man = popup_maneuver(t, X)

    if 5e3 < rm[1]:
        ac = popup_maneuver(t, X, r_targ, ac)

    if norm(ac) > GLIMIT:
        ac = unitize(ac) * GLIMIT
    return ac


def dynamics(t, X, ss, r_targ):
    ac = guidance(t, X, r_targ)
    # ac = [0, 3, 0]
    fd = atmosphere_model(t, X)
    U = np.array([*fd, *ac])
    Xdot = ss.A @ X + ss.B @ U
    return Xdot


# Inits
####################################
t_span = [0, 200]
x0 = np.zeros((12))
x0[:3] = np.array([0, 0, 10])    #R0
x0[3:6] = np.array([0, 200, 0])   #V0

targ_R0 = np.array([0, 50e3, 0])
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

r_pop1 = np.array([0, 7e3, 90])
r_pop2 = np.array([0, 9e3, 10])

# 3D Plot
fig = plt.figure(figsize=(10, 8))
ax = plt.axes(projection='3d')
ax.plot3D(y[:, 0], y[:, 1], y[:, 2])
ax.plot3D(*targ_R0, marker='.')
ax.plot3D(*r_pop1, marker='.', color='green')
ax.plot3D(*r_pop2, marker='.', color='red')
ax.set_xlabel("E")
ax.set_ylabel("N")
ax.set_zlabel("D")

fig2, (ax2, ax3, ax4) = plt.subplots(3, figsize=(10, 8))
ax2.plot(y[:, 1], y[:, 2])
ax2.set_title("y")
ax3.plot(y[:, 1], y[:, 4])
ax3.set_title("yvel")
ax4.plot(y[:, 1], y[:, 5])
ax4.set_title("zvel")

plt.show()

