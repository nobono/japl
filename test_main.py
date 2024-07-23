import unittest
import numpy as np
import quaternion
from japl import Model
from japl import SimObject
from japl import Sim
from japl import AeroTable



class TestExample(unittest.TestCase):

    def __check(self, vehicle: SimObject):
        # for i in vehicle.Y[-1]:
        #     print(i)
        truth = np.array([
            1500.0000000000002,
            0.0,
            10000.003387596103,
            1500.0,
            0.0,
            0.01360428568910825,
            0.0,
            0.008934050694513255,
            0.0,
            0.9999992228258203,
            0.0,
            0.0012467348195054982,
            0.0,
            133.0,
            ])
        self.assertTrue((vehicle.Y[-1] == truth).all())


    def __build_model(self) -> Model:
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

        mass = model.add_state("mass",    13, "mass (kg)")

        Sq = np.array([
            [-q1, -q2, -q3],
            [q0, -q3, q2],
            [q3, q0, -q1],
            [-q2, q1, q0],
            ]) * 0.5

        A = np.array([
            [0,0,0,  1,0,0,  0,0,0,  0,0,0,0,  0], # x
            [0,0,0,  0,1,0,  0,0,0,  0,0,0,0,  0], # y
            [0,0,0,  0,0,1,  0,0,0,  0,0,0,0,  0], # z
            [0,0,0,  0,0,0,  0,0,0,  0,0,0,0,  0], # vx
            [0,0,0,  0,0,0,  0,0,0,  0,0,0,0,  0], # vy
            [0,0,0,  0,0,0,  0,0,0,  0,0,0,0,  0], # vz

            [0,0,0,  0,0,0,  0,0,0,  0,0,0,0,  0], # wx
            [0,0,0,  0,0,0,  0,0,0,  0,0,0,0,  0], # wy
            [0,0,0,  0,0,0,  0,0,0,  0,0,0,0,  0], # wz

            [0,0,0,  0,0,0,  *Sq[0], 0,0,0,0,  0], # q0
            [0,0,0,  0,0,0,  *Sq[1], 0,0,0,0,  0], # q1
            [0,0,0,  0,0,0,  *Sq[2], 0,0,0,0,  0], # q2
            [0,0,0,  0,0,0,  *Sq[3], 0,0,0,0,  0], # q3

            [0,0,0,  0,0,0,  0,0,0,  0,0,0,0,  0], # mass
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

            [0,0,0,  0,0,0],
            ])

        model.ss(A, B)

        return model
        

    def test_example_case1(self):

        model = self.__build_model()
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

        sim.run()

        self.__check(vehicle)
                


if __name__ == '__main__':
    unittest.main()
