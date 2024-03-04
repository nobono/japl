import numpy as np
import control as ct
import matplotlib.pyplot as plt
import scipy
from scipy import constants
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.base import OdeSolver
from autopilot import ss as apss



CD = 0.0002



class ID:
    def __init__(self, state_array: list[str]) -> None:
        assert isinstance(state_array, list)
        for i, state in enumerate(state_array):
            self.__setattr__(state, i)


class State:
    def __init__(self, sol, t) -> None:
        self.t = t
        self.xpos = sol[:, 0]
        self.ypos = sol[:, 1]
        self.xvel = sol[:, 2]
        self.yvel = sol[:, 3]
        self.xacc = sol[:, 4]
        self.yacc = sol[:, 5]
        self.xjerk = sol[:, 6]
        self.yjerk = sol[:, 7]


A = np.array([
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],       # xvel
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],       # yvel
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],       # zvel

    [0, 0, 0, 0, 0, 0, apss.C[0][0], 0, 0, apss.C[0][1], 0, 0], # xacc
    [0, 0, 0, 0, 0, 0, 0, apss.C[0][0], 0, 0, apss.C[0][1], 0], # yacc
    [0, 0, 0, 0, 0, 0, 0, 0, apss.C[0][0], 0, 0, apss.C[0][1]], # zacc

    [0, 0, 0, 0, 0, 0, apss.A[0][0], 0, 0, apss.A[0][1], 0, 0], # xacc_cmd
    [0, 0, 0, 0, 0, 0, 0, apss.A[0][0], 0, 0, apss.A[0][1], 0], # yacc_cmd
    [0, 0, 0, 0, 0, 0, 0, 0, apss.A[0][0], 0, 0, apss.A[0][1]], # zacc_cmd

    [0, 0, 0, 0, 0, 0, apss.A[1][0], 0, 0, apss.A[1][1], 0, 0], # xacc_cmd_dot
    [0, 0, 0, 0, 0, 0, 0, apss.A[1][0], 0, 0, apss.A[1][1], 0], # yacc_cmd_dot
    [0, 0, 0, 0, 0, 0, 0, 0, apss.A[1][0], 0, 0, apss.A[1][1]], # zacc_cmd_dot
    ])

B = np.array([
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [*apss.B[0], 0, 0],
    [0, *apss.B[0], 0],
    [0, 0, *apss.B[0]],
    [*apss.B[1], 0, 0],
    [0, *apss.B[1], 0],
    [0, 0, *apss.B[1]],
    ])

C = np.eye(12)

D = np.zeros((12, 3))

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
    ac[1] = 1*np.sin(0.4 * t)
    return Rt @ ac


def guidance(t, X):
    ucmd = np.array([0, 0, 0])
    # ac = weave_maneuver(t, X)
    ac = popup_maneuver(t, X)
    return ucmd + ac


def dynamics(t, X, ss):
    # pos, vel, ac, ac_dot = X
    ucmd = guidance(t, X)
    U = ucmd
    Xdot = ss.A @ X + ss.B @ U
    # Xdot[5] = -constants.g
    return Xdot


# Inits
####################################
t_span = [0, 40]
x0 = np.zeros((12))
x0[:3] = np.array([0, 0, 4e3])    #R0
x0[3:6] = np.array([0, 2, 0])   #V0
####################################


# Events
####################################
def hit_ground_event(t, y, ss):
    return y[2]
hit_ground_event.terminal = True
####################################


sol = solve_ivp(
        dynamics,
        t_span=t_span,
        t_eval=np.linspace(t_span[0], t_span[1], 1000),
        y0=x0,
        args=(ss,),
        events=[
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

# 3D Plot
fig = plt.figure(figsize=(10, 8))
ax = plt.axes(projection='3d')
ax.plot3D(y[:, 0], y[:, 1], y[:, 2])
ax.set_xlabel("E")
ax.set_ylabel("N")
ax.set_zlabel("D")

# plt.figure()
# plt.plot(t, y[:, 5])
# plt.title("ac")

plt.show()

