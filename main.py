import numpy as np
import control as ct
import matplotlib.pyplot as plt
import scipy
from scipy import constants
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.base import OdeSolver
from tf import ss as apss



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
        # self.xjerk = sol[:, 6]
        # self.yjerk = sol[:, 7]


A = np.array([
    [0, 1, 0, 0],
    [0, 0, *apss.C[0]],
    [0, 0, *apss.A[0]],
    [0, 0, *apss.A[1]],
    ])

B = np.array([
    [0],
    [0],
    apss.B[0],
    apss.B[1],
    ])

C = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    ])

D = np.array([
    [0],
    [0],
    [0],
    [0],
    ])

ss = ct.ss(A, B, C, D)


def autopilot(X, u, ss):
    Xdot = ss.A @ X + ss.B @ u
    return Xdot


def guidance(t, X):
    ucmd = [1]
    return ucmd


def dynamics(t, X, ss):
    pos, vel, ac, ac_dot = X
    ucmd = guidance(t, X)
    U = ucmd

    Xdot = ss.A @ X + ss.B @ U
    # Xdot[1] = -constants.g
    return Xdot


x0 = np.array([0, 0, 0, 0])
t_span = [0, 10]
sol = solve_ivp(dynamics, t_span=t_span, t_eval=np.linspace(*t_span, 1000), y0=x0, args=(ss,))
t = sol['t']
y = sol['y'].T

# state = State(y, t)
# velmag = [scipy.linalg.norm(i) for i in y[:, 2:4]]


fig, (ax, ax2, ax3) = plt.subplots(3, figsize=(10, 8))
ax.plot(t, y[:, 0])
ax.set_title("pos")

ax2.plot(t, y[:, 1])
ax2.set_title("vel")

ax3.plot(t, y[:, 2])
ax3.set_title("ac")

plt.show()



##############################
# x0 = np.array([0, 0])
# u = np.array([1])
# sol = scipy.integrate.odeint(autopilot, x0, t, args=(u, ap_ss_obsv,))
# print(sol)
# t0 = 0.
# u0 = 0.
# y_out = [0.]
# state_out = np.array([x0])
# for _t in t:
#     u = np.sin(_t)
#     tt, yy, state = ct.forced_response(ap_ss, T=[t0, _t], U=[u0, u], X0=x0, return_x=True, transpose=True) #type:ignore
#     y_out += [yy[-1]]
#     state_out = np.append(state_out, [state[-1]], axis=0)
#     t0 = _t
#     u0 = u
#     x0 = state[-1]


# tt, yy, state = ct.forced_response(ap_ss, T=t, U=[0]*100 + [1]*900, X0=x0, return_x=True, transpose=True) #type:ignore
# yy = (ap_ss.C @ sol.T)
# plt.plot(t, sol[:, 0])
# plt.plot(t, sol[:, 1])
# # plt.plot(t, sol[:, 1])
# plt.show()
# quit()
##############################
