import unittest
import numpy as np
from sympy import symbols, Matrix
from japl import Model
from japl.Util import noprint



class TestModel(unittest.TestCase):


    def setup(self):
        pos = Matrix(symbols("x y z"))
        vel = Matrix(symbols("vx vy vz"))
        acc = Matrix(symbols("ax ay az"))
        dt = symbols("dt")
        t = symbols("t")

        state = Matrix([pos, vel])
        input = Matrix([acc])
        X_new = Matrix([
            pos + vel * dt,
            vel + acc * dt,
            ])

        dynamics = X_new.diff(dt)
        return (t, state, input, dt, dynamics)


    @noprint
    def test_has_methods(self):
        t, state, input, dt, dynamics = self.setup()
        model = Model.from_expression(dt, state, input,
                                      dynamics_expr=dynamics,
                                      use_multiprocess_build=False)
        # expr exist
        self.assertEqual(model.has_input_updates_expr(), True)
        self.assertEqual(model.has_state_updates_expr(), True)
        self.assertEqual(model.has_dynamics_expr(), True)
        # Callable functions do not exist yet
        self.assertEqual(model.has_input_updates(), False)
        self.assertEqual(model.has_state_updates(), False)
        self.assertEqual(model.has_dynamics(), False)
        # if we cache_build then functions exist
        model.cache_build(use_parallel=False)
        self.assertEqual(model.has_input_updates(), True)
        self.assertEqual(model.has_state_updates(), True)
        self.assertEqual(model.has_dynamics(), True)


    @noprint
    def test_from_expression(self):
        t, state, input, dt, dynamics = self.setup()
        model = Model.from_expression(dt, state, input, dynamics_expr=dynamics,
                                      use_multiprocess_build=False)
        static = Matrix([])
        self.assertListEqual(list(model.vars), [t, state, input, static, dt])
        self.assertEqual(len(model.state_vars), model.state_dim)
        self.assertEqual(len(model.input_vars), model.input_dim)
        self.assertEqual(model.dynamics_expr, dynamics)
        # self.assertListEqual(model.dynamics(0, [0, 0, 0, 0, 0, 0], [1, 0, 0], [], 0.01).tolist(),
        #                      np.array([0, 0, 0, 1, 0, 0]).tolist())


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

        def pre_update_func(t, X, U, S, dt):
            return []

        def input_func(t, X, U, S, dt):
            return []

        def post_update_func(t, X, U, S, dt):
            return []

        model.pre_update_functions += [pre_update_func]
        model.input_function = input_func
        model.post_update_functions += [post_update_func]
        return model


    @noprint
    def test_from_function(self):
        t, state, input, dt, dynamics = self.setup()

        def func(t, X, U, S, dt):
            subs = {"vx": 0, "vy": 0, "vz": 0, "ax": 1, "ay": 0, "az": 0}
            return np.array(dynamics.subs(subs))

        model = Model.from_function(dt, state, input, dynamics_func=func)
        static = Matrix([])
        self.assertListEqual(list(model.vars), [t, state, input, static, dt])
        self.assertEqual(len(model.state_vars), model.state_dim)
        self.assertEqual(len(model.input_vars), model.input_dim)
        self.assertEqual(model.dynamics, func)
        self.assertListEqual(model.dynamics(0, [0, 0, 0, 0, 0, 0], [1, 0, 0], [], 0.01).flatten().tolist(),
                             np.array([0, 0, 0, 1, 0, 0]).tolist())


    def test_cache_py_function(self):
        model = self.setup_model()
        model.cache_build(use_parallel=False)
        model.dynamics = model.get_cached_function('dynamics')
        self.assertTrue("dynamics" in model._namespace)


    # def test_get_sim_func_call_list(self):
    #     model = self.setup_model()
    #     model.cache_build(use_parallel=False)
    #     call_list = model._get_sim_func_call_list()
    #     self.assertEqual(len(call_list), 6)


if __name__ == '__main__':
    unittest.main()
