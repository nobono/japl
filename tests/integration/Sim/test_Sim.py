import unittest
import numpy as np
from sympy import symbols
from sympy import Matrix
from japl import SimObject
from japl import Model
from japl import Sim
from japl.Util import noprint



class TestSim_integration(unittest.TestCase):


    def setUp(self):
        self.TOLERANCE_PLACES = 15
        self.TOLERANCE = 1e-15


    @noprint
    def setup_model(self):
        dt, pos, vel, acc = symbols("dt, pos, vel, acc")
        state = Matrix([pos, vel])
        input = Matrix([acc])
        pos_new = pos + vel * dt
        vel_new = vel + acc * dt
        state_new = Matrix([pos_new, vel_new])
        dynamics = state_new.diff(dt)  # type:ignore
        model = Model.from_expression(dt_var=dt,
                                      state_vars=state,
                                      input_vars=input,
                                      dynamics_expr=dynamics,  # type:ignore
                                      use_multiprocess_build=False)
        return model


    def test_SimObject_case1(self):
        def input_func(*args, **kwargs):  # type:ignore
            return np.array([1.])
        model = self.setup_model()
        model.set_input_function(input_func)
        model.cache_build(use_parallel=False)
        simobj = SimObject(model)
        simobj.init_state([0, 0])
        sim = Sim(t_span=[0, 1],
                  dt=0.1,
                  simobjs=[simobj],
                  integrate_method="rk4")
        sim.run()
        truth = np.array([[0., 0.],
                          [0.005, 0.1],
                          [0.02, 0.2],
                          [0.045, 0.3],
                          [0.08, 0.4],
                          [0.125, 0.5],
                          [0.18, 0.6],
                          [0.245, 0.7],
                          [0.32, 0.8],
                          [0.405, 0.9],
                          [0.5, 1.]], dtype=float)
        self.assertTrue((simobj.Y - truth).max() < self.TOLERANCE)
        self.assertEqual(sim.istep, sim.Nt)
        self.assertEqual(sim.istep, simobj.get_istep())


    def test_SimObject_connected_models(self):
        def input_func(*args, **kwargs):  # type:ignore # noqa
            return np.array([1.])
        def input_func2(*args, **kwargs):  # type:ignore # noqa
            return np.array([2.])

        # setup model 1 ---------------------------------------------------
        model = self.setup_model()
        model.set_input_function(input_func)
        model.cache_build(use_parallel=False)

        # setup model 2 ---------------------------------------------------
        model2 = self.setup_model()
        model2.set_input_function(input_func2)
        model2.cache_build(use_parallel=False)

        # init simobj 1 ---------------------------------------------------
        simobj_1 = SimObject(model, name="simobj_1")
        simobj_1.init_state([0, 0])

        # init simobj 2 ---------------------------------------------------
        simobj_2 = SimObject(model2, name="simobj_2")
        simobj_2.init_state([0, 0])

        # setup child -----------------------------------------------------
        simobj_1.children_pre_update += [simobj_2]

        # setup & run sim -------------------------------------------------
        sim = Sim(t_span=[0, 1],
                  dt=0.1,
                  simobjs=[simobj_1],
                  integrate_method="rk4")
        sim.run()

        truth_1 = np.array([[0., 0.],
                            [0.005, 0.1],
                            [0.02, 0.2],
                            [0.045, 0.3],
                            [0.08, 0.4],
                            [0.125, 0.5],
                            [0.18, 0.6],
                            [0.245, 0.7],
                            [0.32, 0.8],
                            [0.405, 0.9],
                            [0.5, 1.]], dtype=float)
        truth_2 = [[0., 0.],
                   [0.01, 0.2],
                   [0.04, 0.4],
                   [0.09, 0.6],
                   [0.16, 0.8],
                   [0.25, 1.],
                   [0.36, 1.2],
                   [0.49, 1.4],
                   [0.64, 1.6],
                   [0.81, 1.8],
                   [1., 2.]]
        self.assertTrue((simobj_1.Y - truth_1).max() < self.TOLERANCE)
        self.assertTrue((simobj_2.Y - truth_2).max() < self.TOLERANCE)
        self.assertEqual(simobj_1.get_istep(), simobj_2.get_istep())


if __name__ == '__main__':
    unittest.main()
