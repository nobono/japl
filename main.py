import cProfile
import argparse

import numpy as np
from numpy.linalg import norm

import japl
from japl import Sim
from japl import SimObject
from japl import Model
from japl import AeroTable

# ---------------------------------------------------



japl.set_plotlib("qt")



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


    ###############
    # NOTES: Idea:
    ###############
    # Sim.rules.Talk(send=[simobj], receive=[simobj2, simobj3])
    ###############


    # Model
    ####################################
    model = Model()

    x  = model.add_state("x",         0,  "x (m)")
    y  = model.add_state("y",         1,  "y (m)")
    z  = model.add_state("z",         2,  "z (m)")
    vx = model.add_state("vx",        3,  "xvel (m/s)")
    vy = model.add_state("vy",        4,  "yvel (m/s)")
    vz = model.add_state("vz",        5,  "zvel (m/s)")
    wx = model.add_state("wx",        6,  "wx (rad/s)")
    wy = model.add_state("wy",        7,  "wy (rad/s)")
    wz = model.add_state("wz",        8,  "wz (rad/s)")
    q0 = model.add_state("q0",        9,  "q0")
    q1 = model.add_state("q1",        10, "q1")
    q2 = model.add_state("q2",        11, "q2")
    q3 = model.add_state("q3",        12, "q3")


    Sq = np.array([
        [-q1, -q2, -q3],
        [q0, -q3, q2],
        [q3, q0, -q1],
        [-q2, q1, q0],
        ]) * 0.5

    A = np.array([
        [0,0,0,  1,0,0,  0,0,0,  0,0,0,0], # x
        [0,0,0,  0,1,0,  0,0,0,  0,0,0,0], # y
        [0,0,0,  0,0,1,  0,0,0,  0,0,0,0], # z
        [0,0,0,  0,0,0,  0,0,0,  0,0,0,0], # vx
        [0,0,0,  0,0,0,  0,0,0,  0,0,0,0], # vy
        [0,0,0,  0,0,0,  0,0,0,  0,0,0,0], # vz
                               
        [0,0,0,  0,0,0,  0,0,0,  0,0,0,0], # wx
        [0,0,0,  0,0,0,  0,0,0,  0,0,0,0], # wy
        [0,0,0,  0,0,0,  0,0,0,  0,0,0,0], # wz

        [0,0,0,  0,0,0,  *Sq[0], 0,0,0,0], # q0
        [0,0,0,  0,0,0,  *Sq[1], 0,0,0,0], # q1
        [0,0,0,  0,0,0,  *Sq[2], 0,0,0,0], # q2
        [0,0,0,  0,0,0,  *Sq[3], 0,0,0,0], # q3
        ])

    B = np.array([
        # force  torque
        [0,0,0,  0,0,0],
        [0,0,0,  0,0,0],
        [0,0,0,  0,0,0],
        [1,0,0,  0,0,0],
        [0,1,0,  0,0,0],
        [0,0,1,  0,0,0],

        [0,0,0,  1,0,0],
        [0,0,0,  0,1,0],
        [0,0,0,  0,0,1],

        [0,0,0,  0,0,0],
        [0,0,0,  0,0,0],
        [0,0,0,  0,0,0],
        [0,0,0,  0,0,0],
        ])

    model.ss(A, B)

    vehicle = SimObject(model=model, size=2, color='tab:blue')
    vehicle.aerotable = AeroTable("./aeromodel/aeromodel.pickle")

    vehicle._pre_sim_checks() # TODO move this

    vehicle.plot.set_config({
                "Pos": {
                    "xaxis": "x",
                    "yaxis": "z",
                    },
                "Vel": {
                    "xaxis": "t",
                    "yaxis": "vz",
                    }
                })

    # Inits
    ####################################

    vehicle.Ixx = 1.309 # (kg * m^2)
    vehicle.Iyy = 58.27 # (kg * m^2)
    vehicle.Izz = 58.27 # (kg * m^2)
    vehicle.mass = 133 # (kg)
    vehicle.cg = 1.42 # (m)
    x0 = [0, 0, 0]
    v0 = [20, 0, 30]
    w0 = [0, 0, 0]
    quat0 = [1, 0, 0, 0]
    vehicle.init_state([x0, v0, w0, quat0]) # TODO this should be moved to Model

    # Sim
    ####################################

    sim = Sim(
            t_span=[0, 100],
            dt=.02,
            simobjs=[vehicle],
            events=[],
            animate=1,
            aspect="equal",
            device_input_type="",
            moving_bounds=True,
            rtol=1e-6,
            atol=1e-6,
            blit=False,
            antialias=0,
            figsize=(10, 7),
            body_view=True,
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

