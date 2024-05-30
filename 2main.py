import cProfile
import argparse

# ---------------------------------------------------

import numpy as np
from numpy.linalg import norm

# ---------------------------------------------------

from scipy import constants

# ---------------------------------------------------

from util import unitize
# from util import bound

# ---------------------------------------------------

# from atmosphere import Atmosphere

# ---------------------------------------------------

# from guidance import Guidance

# ---------------------------------------------------

# from maneuvers import Maneuvers

# ---------------------------------------------------

# from control.iosys import StateSpace

# ---------------------------------------------------

from output import OutputManager

# ---------------------------------------------------

from japl.Sim.Sim import Sim
from japl.SimObject.SimObject import SimObject
from japl.Model.Model import Model

# ---------------------------------------------------



# def atmosphere_drag(rm, vm):
#     ATMOS_BOUNDS = [-5e3, 81e3]
#     alt_bounded = bound(rm[2] / 1000, *ATMOS_BOUNDS)
#     density = atmosphere.density(alt_bounded)
#     CD = 0.45
#     A = .25**2
#     MASS = 1.0
#     xfd = -(0.5 * CD * A * density * vm[0]) / MASS
#     yfd = -(0.5 * CD * A * density * vm[1]) / MASS
#     zfd = -(0.5 * CD * A * density * vm[2]) / MASS
#     return np.array([xfd, yfd, zfd])


# def atmosphere_gravity(rm):
#     ATMOS_BOUNDS = [-5e3, 81e3]
#     alt_bounded = bound(rm[2] / 1000, *ATMOS_BOUNDS)
#     grav_accel = atmosphere.grav_accel(alt_bounded)
#     return grav_accel


# def guidance_func(t, rm, vm, r_targ, config):
#     GLIMIT = float(config.get("GLIMIT", 14.0))

#     ac = guidance.run(t, rm, vm, r_targ, config)

#     if (norm(ac) - (GLIMIT * constants.g)) > 1e-10:
#         ac = unitize(ac) * (GLIMIT * constants.g)
#     return ac


def dynamics_func(t, X, simobj: SimObject):
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
    Xdot = simobj.step(X, U)
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

    # Model
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

    model = Model.from_statespace(A, B, C, D)
    vehicle = SimObject(model=model)

    vehicle.register_state("x",         0, "x (m)")
    vehicle.register_state("y",         1, "y (m)")
    vehicle.register_state("z",         2, "z (m)")
    vehicle.register_state("vx",        3, "xvel (m/s)")
    vehicle.register_state("vy",        4, "yvel (m/s)")
    vehicle.register_state("vz",        5, "zvel (m/s)")
    vehicle.register_state("fuel_burn", 6, "Fuel Burn ")

    # Inits
    ###############

    x0 = [0, 0, 0]
    v0 = [0, 0, 0]
    vehicle.init_state([x0, v0, 0])

    # ss = StateSpace(A, B, C, D)
    # atmosphere = Atmosphere()
    # guidance = Guidance()
    # maneuver = Maneuvers()

    Sim(t_span=[0, 100], dt=0.01, simobjs=[vehicle])()

    # r_pop1 = np.array([0, 47e3, 90])
    # r_pop2 = np.array([0, 45e3, 10])
    config = {
            "plot": {
                "XY": {
                    "x_axis": "x",
                    "y_axis": "y",
                    },
                "Vel": {
                    "x_axis": "time",
                    "y_axis": "vy",
                    },
                "Fuel Burn": {
                    "x_axis": "time",
                    "y_axis": "fuel_burn",
                    }
                }
            }

    T = vehicle.T
    Y = vehicle.Y
    plot_points = []
    plot_config = config.get("plot", {})
    output_manager = OutputManager(vehicle, args, plot_config, T, Y, plot_points, figsize=(10, 8))
    output_manager.plots()

