import cProfile
import argparse

import numpy as np
from numpy.linalg import norm

from scipy import constants

from util import unitize

from output import OutputManager

import japl
from japl import Sim
from japl import SimObject
from japl import Model

from events import hit_ground_event

# ---------------------------------------------------



japl.set_plotlib("qt")


def dynamics_func(t, X, simobj: SimObject):

    ac = np.array([0, 5, 0])

    fuel_burn = X[6]
    if fuel_burn >= 100:
        ac = np.zeros((3,))

    burn_const = 0.2

    U = np.array([*ac])
    Xdot = simobj.step(X, U)
    Xdot[6] = burn_const * norm(ac) #type:ignore
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
    parser.add_argument(
        "--figsize",
        dest="figsize",
        type=float,
        nargs="*",
        default=(6, 4),
    )
    args = parser.parse_args()
    args.plot = True


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

    model = Model.ss(A, B)
    vehicle = SimObject(model=model, size=2, color='tab:blue')

    vehicle.register_state("x",         0, "x (m)")
    vehicle.register_state("y",         1, "y (m)")
    vehicle.register_state("z",         2, "z (m)")
    vehicle.register_state("vx",        3, "xvel (m/s)")
    vehicle.register_state("vy",        4, "yvel (m/s)")
    vehicle.register_state("vz",        5, "zvel (m/s)")
    vehicle.register_state("fuel_burn", 6, "Fuel Burn ")

    vehicle.plot.set_config({
                "Pos": {
                    "xaxis": "x",
                    "yaxis": "z",
                    },
                "Vel": {
                    "xaxis": "x",
                    "yaxis": "vx",
                    }
                })

    # Inits
    ####################################

    x0 = [0, 0, 0]
    v0 = [20, 0, 30]
    vehicle.init_state([x0, v0, 0])

    # Sim
    ####################################

    sim = Sim(
            t_span=[0, 100],
            dt=.01,
            simobjs=[vehicle],
            events=[],
            animate=1,
            aspect="equal",
            device_input_type="gamepad",
            moving_bounds=True,
            rtol=1e-6,
            atol=1e-6,
            blit=False,
            antialias=False,
            figsize=(10, 7)
            )
    sim.run()

    # config = {
    #         "plot": {
    #             "XY": {
    #                 "x_axis": "x",
    #                 "y_axis": "y",
    #                 },
    #             "Vel": {
    #                 "x_axis": "time",
    #                 "y_axis": "vy",
    #                 },
    #             "Fuel Burn": {
    #                 "x_axis": "time",
    #                 "y_axis": "fuel_burn",
    #                 }
    #             }
    #         }

    # T = sim.T
    # Y = vehicle.Y
    # plot_points = []
    # plot_config = config.get("plot", {})
    # output_manager = OutputManager(vehicle, args, plot_config, T, Y, plot_points, figsize=(8, 6))
    # output_manager.plots()

