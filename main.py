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

import guidance

# ---------------------------------------------------

from maneuvers import Maneuvers

# ---------------------------------------------------

from dynamics import FirstOrderInput
from dynamics import SecondOrderInput

# ---------------------------------------------------

from output import OutputManager

# ---------------------------------------------------

from events import hit_ground_event
from events import hit_target_event
from events import check_for_events

# ---------------------------------------------------


ss = FirstOrderInput()
atmosphere = Atmosphere()
maneuvers = Maneuvers()



def atmosphere_model(rm, vm):
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


def guidance_func(t, rm, vm, r_targ):
    # ac = guidance_func(rm, vm, r_targ)
    # ac = maneuvers.popup(rm, vm, r_targ)
    # ac = maneuvers.uo_dive(rm, vm, r_targ)
    ac = maneuvers.test(t, rm, vm, r_targ)
    return ac


def dynamics_func(t, X, ss, r_targ):
    rm = X[:3]
    vm = X[3:6]
    ac = guidance_func(t, rm, vm, r_targ)
    a_drag = np.zeros((3,)) #atmosphere_model(rm, vm)
    U = np.array([*a_drag, *ac])
    Xdot = ss.A @ X + ss.B @ U
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
    )
    args = parser.parse_args()

    # Load config file
    config = read_config_file("template.yaml")

    # Inits
    ####################################
    t_span = [0, 300]
    dt = 0.01

    x0 = ss.get_init_state()
    x0[:3] = np.array([0, 50e3, 10])    #R0
    x0[3:6] = np.array([0, -200, 0])   #V0

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
            args=(ss, targ_R0),
            events=[
                hit_target_event,
                hit_ground_event,
                ],
            rtol=1e-3,
            atol=1e-6,
            max_step=0.2,
            )
    T = sol['t']
    Y = sol['y'].T
    ##############################
     
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
    OutputManager(args, T, Y, plot_points).plots(x_axis='t')
