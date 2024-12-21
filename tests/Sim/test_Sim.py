import unittest
import numpy as np
from japl.Sim.Sim import Sim



class TestSim(unittest.TestCase):


    def setUp(self):
        self.TOLERANCE_PLACES = 15
        self.TOLERANCE = 1e-15


    def test_init(self):
        sim = Sim(t_span=[0, 1],
                  dt=0.1,
                  simobjs=[],
                  integrate_method="rk4")
        self.assertEqual(sim.t_span, [0, 1])
        self.assertEqual(sim.dt, 0.1)
        self.assertEqual(sim.simobjs, [])
        self.assertEqual(sim.integrate_method, 'rk4')
        self.assertEqual(sim.istep, 0)
        self.assertEqual(sim.Nt, 10)
        self.assertListEqual(sim.t_array.tolist(), np.linspace(0, 1, 10 + 1).tolist())
        self.assertListEqual(sim.T.tolist(), [0.] * 11)


if __name__ == '__main__':
    unittest.main()
