import cProfile
import os
import sys
from tqdm import tqdm
lib_path = os.path.join(os.getcwd(), "lib")
sys.path.append(lib_path)

# ---------------------------------------------------

import argparse
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

# ---------------------------------------------------

from scipy.integrate import solve_ivp
from scipy import constants

# ---------------------------------------------------

from util import unitize
from util import bound
from util import create_C_rot
from util import inv
from util import read_config_file
from util import norm

# ---------------------------------------------------

from atmosphere import Atmosphere

# ---------------------------------------------------

from guidance import Guidance

# ---------------------------------------------------

from maneuvers import Maneuvers

# ---------------------------------------------------

from dynamics import FirstOrderInput
from dynamics import SecondOrderInput
from control.iosys import StateSpace

# ---------------------------------------------------

from output import OutputManager

# ---------------------------------------------------

from events import hit_ground_event
from events import hit_target_event
from events import check_for_events

# ---------------------------------------------------

# for eval
sin = np.sin
cos = np.cos
tan = np.tan
atan = np.arctan
atan2 = np.arctan2
acos = np.arccos
asin = np.arcsin
degrees = np.degrees
radians = np.radians
pi = np.pi



def atmosphere_drag(rm, vm):
    ATMOS_BOUNDS = [-5e3, 81e3]
    alt_bounded = bound(rm[2] / 1000, *ATMOS_BOUNDS)
    density = atmosphere.density(alt_bounded)
    CD = 0.45
    A = .25**2
    MASS = 1.0
    xfd = -(0.5 * CD * A * density * vm[0]) / MASS
    yfd = -(0.5 * CD * A * density * vm[1]) / MASS
    zfd = -(0.5 * CD * A * density * vm[2]) / MASS
    return np.array([xfd, yfd, zfd])


def atmosphere_gravity(rm):
    ATMOS_BOUNDS = [-5e3, 81e3]
    alt_bounded = bound(rm[2] / 1000, *ATMOS_BOUNDS)
    grav_accel = atmosphere.grav_accel(alt_bounded)
    return grav_accel


def guidance_func(t, rm, vm, r_targ, config):
    GLIMIT = float(config.get("GLIMIT", 14.0))

    ac = guidance.run(t, rm, vm, r_targ, config)

    if (norm(ac) - (GLIMIT * constants.g)) > 1e-10:
        ac = unitize(ac) * (GLIMIT * constants.g)
    return ac


def dynamics_func(t, X, ss, r_targ, config):
    # rm = X[:3]
    # vm = X[3:6]
    # atmos_drag_enable = config["guidance"]["phase"][guidance.phase_id].get("enable_drag", False)
    # gravity_enable = config["guidance"]["phase"][guidance.phase_id].get("enable_gravity", False)

    # ac = guidance_func(t, rm, vm, r_targ, config)
    # ac = np.zeros((3,))

    ac = np.array([0, 5, 0])

    fuel_burn = X[6]
    if fuel_burn >= 100:
        ac = np.zeros((3,))

    # External Accelerations
    # acc_ext = np.zeros((3,))
    # if atmos_drag_enable:
    #     acc_ext += atmosphere_drag(rm, vm)
    # if gravity_enable:
    #     acc_ext += atmosphere_gravity(rm) * np.array([0, 0, -1])

    burn_const = 0.2

    U = np.array([*ac])
    Xdot = ss.A @ X + ss.B @ U
    Xdot[6] = burn_const * norm(ac)
    return Xdot



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
    parser.add_argument(
        "-s",
        dest="save",
        action="store_true",
    )
    parser.add_argument(
        "-i",
        dest="input",
        type=str,
        default="",
    )
    args = parser.parse_args()
    args.plot = True

    # Load config file
    # config = {}
    # if args.input:
    #     config = read_config_file(args.input)

    config = {
            "plot": {
                "XY": {
                    "x_axis": "north",
                    "y_axis": "east",
                    },
                "Vel": {
                    "x_axis": "north",
                    "y_axis": "north_dot",
                    }
                }
            }

    # Inits
    ####################################

    A = np.array([
        [0, 0, 0, 1, 0, 0,  0],
        [0, 0, 0, 0, 1, 0,  0],
        [0, 0, 0, 0, 0, 1,  0],
        [0, 0, 0, 0, 0, 0,  0],
        [0, 0, 0, 0, 0, 0,  0],
        [0, 0, 0, 0, 0, 0,  0],

        [0, 0, 0, 0, 0, 0,  1],
        ])
    B = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],

        [0, 0, 0],
        ])
    C = np.eye(len(A))
    D = np.zeros(B.shape)

    ss = StateSpace(A, B, C, D)
    atmosphere = Atmosphere()
    guidance = Guidance()
    maneuver = Maneuvers()

    t_span = [0, 100]
    dt = 0.01

    R0 = np.array([0, 0, 0])
    V0 = np.array([0, 0, 0])
    x0 = np.concatenate([R0, V0, [0]])

    targ_R0 = np.array([0, 0, 0])
    ####################################

    t_array = np.linspace(t_span[0], t_span[1], int(t_span[1]/dt))
    T = np.zeros(t_array.shape)
    Y = np.zeros((t_array.shape[0], x0.shape[0]))
    T[0] = t_span[0]
    Y[0] = x0

    ##############################
    sol = solve_ivp(
            dynamics_func,
            t_span=t_span,
            t_eval=t_array,
            y0=x0,
            args=(ss, targ_R0, config),
            events=[
                # hit_target_event,
                # hit_ground_event,
                ],
            rtol=1e-3,
            atol=1e-6,
            max_step=0.2,
            )
    T = sol['t']
    Y = sol['y'].T

    ##############################

    # plot fuel burn
    plt.figure()
    plt.plot(T, Y[:, 6])
     
    # for istep, (tstep_prev, tstep) in tqdm(enumerate(zip(t_array, t_array[1:])),
    #                                        total=len(t_array)):

    #     sol = solve_ivp(
    #             dynamics_func,
    #             t_span=(tstep_prev, tstep),
    #             t_eval=[tstep],
    #             y0=x0,
    #             args=(ss, targ_R0),
    #             events=[
    #                 hit_target_event,
    #                 hit_ground_event,
    #                 ],
    #             rtol=1e-3,
    #             atol=1e-6,
    #             )

    #     # check for stop event
    #     if check_for_events(sol['t_events']):
    #         # truncate output arrays if early stoppage
    #         T = T[:istep + 1]
    #         Y = Y[:istep + 1]
    #         break
    #     else:
    #         # store output
    #         t = sol['t'][0]
    #         y = sol['y'].T[0]
    #         T[istep + 1] = t
    #         Y[istep + 1] = y
    #         x0 = Y[istep + 1]

    # r_pop1 = np.array([0, 47e3, 90])
    # r_pop2 = np.array([0, 45e3, 10])
    plot_points = []
    plot_config = config.get("plot", {})
    OutputManager(args, plot_config, T, Y, plot_points, figsize=(6, 5)).plots()
