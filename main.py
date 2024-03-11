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

from guidance import Guidance

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
guidance = Guidance()
maneuver = Maneuvers()



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


def guidance_func(t, rm, vm, r_targ, config):
    GLIMIT = 14.0
    if "guidance" in config:
        gd_phase = config["guidance"]["phase"][guidance.phase_id]
        gd_type = gd_phase["type"]
        gd_name = gd_phase["name"]
        gd_args = gd_phase["args"]
        gd_condition_next = gd_phase.get("condition_next", None)

        gd_obj = globals().get(gd_type)
        gd_func = gd_obj.__getattribute__(gd_name)

        # State pkg
        range = norm(rm)
        east = rm[0]
        north = rm[1]
        alt = rm[2]
        east_dot = vm[0]
        north_dot = vm[1]
        alt_dot = vm[2]
        state = {
                "rm": rm,
                "vm": vm,
                "range": range,
                "alt": alt,
                "alt_dot": alt_dot,
                "north": north,
                "east": east,
                "north_dot": north_dot,
                "east_dot": east_dot,
                }
        ac = gd_func(t, state, gd_args, r_targ=r_targ)

        if gd_condition_next and eval(gd_condition_next):
            # check if next phase is defined
            # else hold current phase
            next_phase = guidance.phase_id + 1
            if next_phase in config["guidance"]["phase"]:
                guidance.phase_id += 1

    # ac = np.zeros((3,))
    #####################
    # ac = guidance_func(rm, vm, r_targ)
    # ac = maneuver.popup(rm, vm, r_targ)
    # ac = maneuver.uo_dive(rm, vm, r_targ)
    # ac = maneuver.test(t, rm, vm, r_targ, config)
    # ac = maneuver.weave_maneuver(t, vm)
    # ac = maneuver.climb(t, rm, vm, r_targ, config)
    # vd = np.array([0, -1, 50 * np.sin(0.1 * t)])
    ########################
    # vd = np.array([np.sin(0.1 * t), -1, np.cos(0.1 * t)])
    # ac = guidance.PN(vd, vm, 1,
    #                  bounds=[-(GLIMIT * constants.g), (GLIMIT * constants.g)])
    ################################
    # alt_des = 50 * np.sin(.2 * t) + rm[2]
    # ALT_RATE_LIMIT = 1000.0
    # ALT_TIME_CONST = 10.0 # (s)
    # KP = 1.0 / ALT_TIME_CONST
    # KD = 0.7
    # alt = rm[2]
    # alt_dot = vm[2]
    # C_i_v = create_C_rot(vm)
    # bounds = [-ALT_RATE_LIMIT, ALT_RATE_LIMIT]
    # ac_alt = guidance.pd_controller(alt_des, alt, alt_dot, KP, KD, bounds=bounds)
    # ac = C_i_v @ np.array([0, 0, ac_alt])
    # #####
    # x_des = 50 * np.cos(.2 * t) + rm[0]
    # ALT_RATE_LIMIT = 1000.0
    # ALT_TIME_CONST = 10.0 # (s)
    # KP = 1.0 / ALT_TIME_CONST
    # KD = 0.7
    # x = rm[0]
    # x_dot = vm[0]
    # C_i_v = create_C_rot(vm)
    # bounds = [-ALT_RATE_LIMIT, ALT_RATE_LIMIT]
    # x_alt = guidance.pd_controller(x_des, x, x_dot, KP, KD, bounds=bounds)
    # ac = ac + C_i_v @ np.array([0, x_alt, 0])
    ################################

    if (norm(ac) - (GLIMIT * constants.g)) > 1e-8:
        ac = unitize(ac) * (GLIMIT * constants.g)
    return ac


def dynamics_func(t, X, ss, r_targ, config):
    rm = X[:3]
    vm = X[3:6]
    ac = guidance_func(t, rm, vm, r_targ, config)
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
        default="",
    )
    args = parser.parse_args()

    # Load config file
    config = {}
    if args.input:
        config = read_config_file(args.input)

    # Inits
    ####################################
    # t_span = [0, 5]
    # dt = 0.01
    init = config.get('init', None)
    t_span = init.get("t_span", [0, 200])
    dt = init.get("dt", 0.01)

    x0 = ss.get_init_state()
    # x0[:3] = np.array([0, 50e3, 10])    #R0
    # x0[3:6] = np.array([0, -200, 0])   #V0
    x0[:3] = config.get("R0", np.array([0, 50e3, 10]))
    x0[3:6] = config.get("V0", np.array([0, -200, 0]))

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
    OutputManager(args, T, Y, plot_points).plots(x_axis='x')
