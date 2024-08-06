import unittest
import numpy as np
import quaternion
from japl import Model
from japl import SimObject
from japl import Sim
from japl import AeroTable
from sympy import MatrixSymbol, Matrix, symbols
from japl.Library.Vehicles import MissileGeneric



class test_MissileGeneric(unittest.TestCase):


    def setUp(self):
        self.TOLERANCE_PLACES = 14

        model = MissileGeneric.model
        self.vehicle = SimObject(model=model, size=2, color='tab:blue')
        self.vehicle.aerotable = AeroTable("./aeromodel/aeromodel_psb.mat")
        self.vehicle.Ixx = 1.309 # (kg * m^2)
        self.vehicle.Iyy = 58.27 # (kg * m^2)
        self.vehicle.Izz = 58.27 # (kg * m^2)
        self.vehicle.mass = 133 # (kg)
        self.vehicle.cg = 1.42 # (m)
        x0 = [0, 0, 10000]
        v0 = [1500, 0, 0]
        w0 = [0, 0, 0]
        quat0 = quaternion.from_euler_angles([0, 0, 0]).components
        mass0 = 133.0
        gravity0 = [0, 0, -9.81]
        speed0 = np.linalg.norm(v0)
        self.vehicle.init_state([x0, v0, w0, quat0, mass0, gravity0, speed0]) # TODO this should be moved to Model


    def test_MissileGeneric_case1(self):
        sim = Sim(
                t_span=[0, 0.1],
                dt=.01,
                simobjs=[self.vehicle],
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
        sim.run()

        truth = [150.00000000000002842171,
                 0.00000000000000000000,
                 9999.95115103819807700347,
                 1500.00000000000000000000,
                 0.00000000000000000000,
                 -0.97676241029288801698,
                 0.00000000000000000000,
                 0.00058360852455692492,
                 0.00000000000000000000,
                 0.99999999993436439194,
                 0.00000000000000000000,
                 0.00001145716553824678,
                 0.00000000000000000000,
                 133.00000000000000000000,
                 0.00000000000000000000,
                 0.00000000000000000000,
                 -9.77586844288743428422]
        for state, tru in zip(self.vehicle.Y[-1], truth):
            self.assertAlmostEqual(state, tru, places=self.TOLERANCE_PLACES)



if __name__ == '__main__':
    unittest.main()
