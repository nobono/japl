import cProfile
import argparse

import numpy as np
from numpy.linalg import norm
import quaternion
from sympy import Matrix, symbols

import japl
from japl import Sim
from japl import SimObject
from japl import Model
from japl import AeroTable

from japl.Library.Vehicles import RigidBodyModel
from japl.Library.Vehicles import MissileGeneric_example

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
    model = MissileGeneric_example.model
    vehicle = SimObject(model=model, size=2, color='tab:blue')
    vehicle.aerotable = AeroTable("./aeromodel/aeromodel_psb.mat")

    vehicle.plot.set_config({
                "Pos": {
                    "xaxis": "pos_x",
                    "yaxis": "pos_z",
                    "aspect": "auto",
                    },
                "Vel": {
                    "xaxis": "t",
                    "yaxis": "vel_z",
                    "aspect": "auto",
                    },
                })

    # Inits
    ####################################

    vehicle.Ixx = 1.309 # (kg * m^2)
    vehicle.Iyy = 58.27 # (kg * m^2)
    vehicle.Izz = 58.27 # (kg * m^2)
    vehicle.mass = 133 # (kg)
    vehicle.cg = 1.42 # (m)
    x0 = [0, 0, 10000]
    v0 = [1500, 0, 0]
    w0 = [0, 0, 0]
    quat0 = quaternion.from_euler_angles([0, 0, 0]).components
    mass0 = 133.0
    gravity0 = [0, 0, -9.81]
    speed0 = 1500
    vehicle.init_state([x0, v0, w0, quat0, mass0, gravity0, speed0]) # TODO this should be moved to Model

    # Sim
    ####################################

    # TODO dt is refresh rate for animation
    # but dt just create t_array for no animation
    sim = Sim(
            t_span=[0, 0.1],
            dt=.01,
            simobjs=[vehicle],
            integrate_method="rk4",
            events=[],
            aspect="equal",
            device_input_type="",
            moving_bounds=True,
            rtol=1e-6,
            atol=1e-6,
            blit=False,
            antialias=0,
            figsize=(10, 7),
            instrument_view=1,
            draw_cache_mode=0,
            animate=0,
            frame_rate=25,
            quiet=1,
            )

    # sim.plotter.add_text("debug")
    # sim.plotter.add_text("ytorque")
    # sim.plotter.add_text("iota")
    sim.run()

    # speed = vehicle.get_state_array(vehicle.Y[-1], "speed")
    # vel_norm = np.linalg.norm(vehicle.Y[-1][3:6])
    # quat = vehicle.get_state_array(vehicle.Y[-1], ["q_0", "q_1", "q_2", "q_3"])
    # print(vel_norm)
    # print(vel_norm - speed)
    # for i in vehicle.Y[-1]:
    #     print("%.20f," % i)


    # plt.plot(np.degrees(alpha), CN)

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

