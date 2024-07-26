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
    pos = Matrix(symbols("x y z"))      # must be fixed for AeroModel
    vel = Matrix(symbols("vx vy vz"))   # must be fixed for AeroModel
    acc = Matrix(symbols("ax ay az"))
    tq = Matrix(symbols("tqx tqy tqz"))
    w = Matrix(symbols("wx wy wz"))
    q = Matrix(symbols("q0 q1 q2 q3"))  # must be fixed for AeroModel

    dt = symbols("dt")
    mass = symbols("mass")

    w_skew = Matrix(w).hat()        #type:ignore
    Sw = Matrix(np.zeros((4,4)))
    Sw[0, :] = Matrix([0, *w]).T
    Sw[:, 0] = Matrix([0, *-w])     #type:ignore
    Sw[1:, 1:] = w_skew

    x_new = pos + vel * dt
    v_new = vel + acc * dt
    w_new = w + tq * dt
    q_new = q + (-0.5 * Sw * q) * dt
    mass_new = mass

    X_new = Matrix([
        x_new.as_mutable(),
        v_new.as_mutable(),
        w_new.as_mutable(),
        q_new.as_mutable(),
        mass_new,
        ])

    state = Matrix([pos, vel, w, q, mass])
    input = Matrix([acc, tq])

    dynamics: Matrix = X_new.diff(dt) #type:ignore

    model = Model().from_expression(dt, state, input, dynamics)

    vehicle = SimObject(model=model, size=2, color='tab:blue')
    vehicle.aerotable = AeroTable("./aeromodel/aeromodel_psb.mat")

    vehicle.plot.set_config({
                "Pos": {
                    "xaxis": "x",
                    "yaxis": "z",
                    "aspect": "auto",
                    },
                "Vel": {
                    "xaxis": "t",
                    "yaxis": "vz",
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
    vehicle.init_state([x0, v0, w0, quat0, mass0]) # TODO this should be moved to Model

    # Sim
    ####################################

    # TODO dt is refresh rate for animation
    # but dt just create t_array for no animation
    sim = Sim(
            t_span=[0, 1],
            dt=.01,
            simobjs=[vehicle],
            integrate_method="rk4",
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
            instrument_view=1,
            draw_cache_mode=0,
            quiet=1, # TODO still working on this
            )

    # sim.plotter.add_text("debug")
    # sim.plotter.add_text("ytorque")
    # sim.plotter.add_text("iota")
    sim.run()


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

