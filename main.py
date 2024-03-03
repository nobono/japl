import numpy as np
import control as ct
import matplotlib.pyplot as plt
import scipy




g = -9.81
CD = 0.0002


class State:
    def __init__(self, sol, t) -> None:
        self.t = t
        self.xpos = sol[:, 0]
        self.ypos = sol[:, 1]
        self.xvel = sol[:, 2]
        self.yvel = sol[:, 3]


A = np.array([
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    ])

B = np.array([
    [0, 0],
    [0, 0],
    [1, 0],
    [0, 1],
    ])

C = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    ])

D = np.array([
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    ])

ss = ct.ss(A, B, C, D)

#########
wapar = 5
zetapar = .6
# ap_tf = ct.tf([1], [1/wapar**2, 2*zetapar/wapar, 1])
ap_ss = ct.tf2ss([1], [1/wapar**2, 2*zetapar/wapar, 1])
ap_ss_obsv, TM = ct.observable_form(ap_ss)
am_perp_ss = ct.parallel(ap_ss, ap_ss, ap_ss)
#########

def ss_dynamics(X, t, u, ss):
    u = np.array([0])
    Xdot = ss.A @ X + ss.B @ u
    return Xdot


def autopilot(X, t, u, ss):
    Xdot = ss.A @ X + ss.B @ u
    return Xdot


def guidance(X, t):
    xpos, ypos, xvel, yvel = X
    # u = [1, 1]
    ucmd = [0, 0]
    ucmd[1] = -np.sin(0.1 * t)
    return ucmd


def dynamics(X, t, ss):
    xpos, ypos, xvel, yvel = X
    ucmd = guidance(X, t)
    U = np.array([0, np.sin(0.1 * t)])
    y_axis_ap = np.array([0, 0])
    ap_ret = autopilot(y_axis_ap, t, [ucmd[1]], ap_ss_obsv)
    U[1] = ap_ss.C @ ap_ret.T
    # U[0] += -(CD * xvel**2)
    # U[1] += g - (CD * yvel**2)
    Xdot = ss.A @ X + ss.B @ U
    return Xdot


t = np.linspace(0, 10, 1000)

##############################
x0 = np.array([0, 0])
u = np.array([1])
sol = scipy.integrate.odeint(autopilot, x0, t, args=(u, ap_ss_obsv,))
print(sol)
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
plt.plot(t, sol[:, 0])
plt.plot(t, sol[:, 1])
# # plt.plot(t, sol[:, 1])
plt.show()
quit()
##############################


x0 = np.array([0, 0, 2, 0])
sol = scipy.integrate.odeint(dynamics, x0, t, args=(ss,))
state = State(sol, t)
velmag = [scipy.linalg.norm(i) for i in sol[:, 2:4]]


fig, (ax, ax2) = plt.subplots(2, figsize=(10, 8))
ax.plot(state.t, state.ypos)
ax.plot(state.t, [100 * np.sin(0.1 * i) for i in state.t], '--')
ax.set_title("XY")

ax2.plot(state.t, state.yvel)
ax2.set_title("Vel Mag")

# ax.plot(t, sol[:, 0])

plt.show()



