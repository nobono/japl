import cProfile
import os
import sys
lib_path = os.path.join(os.getcwd(), "lib")
sys.path.append(lib_path)

# ---------------------------------------------------

import argparse
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------

from scipy.linalg import norm
from scipy.integrate import solve_ivp

# ---------------------------------------------------

from util import unitize
from util import inv
from util import bound

# ---------------------------------------------------

from atmosphere import Atmosphere

# ---------------------------------------------------

import guidance
from maneuvers import Maneuvers

# ---------------------------------------------------

from dynamics import ss

# ---------------------------------------------------



atmosphere = Atmosphere()
maneuvers = Maneuvers()



def atmosphere_model(t, X):
    rm = X[:3]
    vm = X[3:6]
    ATMOS_BOUNDS = [-5e3, 81e3]
    alt_bounded = bound(rm[2] / 1000, *ATMOS_BOUNDS)
    density = atmosphere.density(alt_bounded)
    CD = 0.45
    A = .25**2
    xfd = -(0.5 * CD * A * density * vm[0])
    yfd = -(0.5 * CD * A * density * vm[1])
    zfd = -(0.5 * CD * A * density * vm[2])
    return np.array([xfd, yfd, zfd])


def guidance_func(t, X, r_targ):
    GLIMIT = 14.0
    rm = X[:3]
    vm = X[3:6]
    ac = unitize(vm) * 3.5
    # ac = weave_maneuver(t, X)
    # ac_man = popup_maneuver(t, X)

    if 5e3 < rm[1]:
        ac = maneuvers.popup_maneuver(t, X, r_targ, ac)

    if norm(ac) > GLIMIT:
        ac = unitize(ac) * GLIMIT
    return ac


def dynamics_func(t, X, ss, r_targ):
    ac = guidance_func(t, X, r_targ)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--3d",
        dest="plot_3d",
        action="store_true",
        )
    parser.add_argument(
        "-p",
        dest="plot",
        action="store_true",
    )
    args = parser.parse_args()

    sol = solve_ivp(
            dynamics_func,
            t_span=t_span,
            t_eval=np.linspace(t_span[0], t_span[1], 5000),
            y0=x0,
            args=(ss, targ_R0),
            events=[
                hit_target_event,
                hit_ground_event,
                ]
            )

    t = sol['t']
    y = sol['y'].T

    # velmag = [scipy.linalg.norm(i) for i in y[:, 2:4]]

    r_pop1 = np.array([0, 7e3, 90])
    r_pop2 = np.array([0, 9e3, 10])

    if args.plot_3d:
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

    if args.plot:
        fig2, (ax2, ax3, ax4) = plt.subplots(3, figsize=(10, 8))
        ax2.plot(y[:, 1], y[:, 2])
        ax2.set_title("y")
        ax3.plot(y[:, 1], y[:, 4])
        ax3.set_title("yvel")
        ax4.plot(y[:, 1], y[:, 5])
        ax4.set_title("zvel")

    plt.show()

