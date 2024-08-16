import unittest
import numpy as np
from sympy import symbols, Matrix
from japl import Model



class TestExample(unittest.TestCase):


    def setup(self):
        pos = Matrix(symbols("x y z"))
        vel = Matrix(symbols("vx vy vz"))
        acc = Matrix(symbols("ax ay az"))
        dt = symbols("dt")

        state = Matrix([pos, vel])
        input = Matrix([acc])
        X_new = Matrix([
            pos + vel * dt,
            vel + acc * dt,
            ])

        dynamics = X_new.diff(dt)
        return (state, input, dt, dynamics)


    def test_from_function(self):
        state, input, dt, dynamics = self.setup()

        def func(X, U, dt):
            subs = {"vx": 0, "vy": 0, "vz": 0, "ax": 1, "ay": 0, "az": 0}
            return np.array(dynamics.subs(subs))

        model = Model.from_function(dt, state, input, func)
        self.assertListEqual(list(model.vars), [state, input, dt])
        self.assertEqual(len(model.state_vars), model.state_dim)
        self.assertEqual(len(model.input_vars), model.input_dim)
        self.assertEqual(model.dynamics_func, func)
        self.assertListEqual(model([0, 0, 0, 0, 0, 0], [1, 0, 0], 0.01).tolist(), np.array([0, 0, 0, 1, 0, 0]).tolist())


    def test_from_statespace(self):
        state, input, dt, dynamics = self.setup()
        A = np.array([
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            ])
        B = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            ])
        model = Model.from_statespace(dt, state, input, A, B)
        self.assertListEqual(list(model.vars), [state, input, dt])
        self.assertEqual(len(model.state_vars), model.state_dim)
        self.assertEqual(len(model.input_vars), model.input_dim)
        self.assertListEqual(model.A.tolist(), A.tolist())
        self.assertListEqual(model.B.tolist(), B.tolist())
        self.assertListEqual(model([0, 0, 0, 0, 0, 0], [1, 0, 0], 0.01).tolist(), np.array([0, 0, 0, 1, 0, 0]).tolist())


    def test_from_expression(self):
        state, input, dt, dynamics = self.setup()
        model = Model.from_expression(dt, state, input, dynamics)
        self.assertListEqual(list(model.vars), [state, input, dt])
        self.assertEqual(len(model.state_vars), model.state_dim)
        self.assertEqual(len(model.input_vars), model.input_dim)
        self.assertEqual(model.dynamics_expr, dynamics)
        self.assertListEqual(model([0, 0, 0, 0, 0, 0], [1, 0, 0], 0.01).tolist(), np.array([0, 0, 0, 1, 0, 0]).tolist())


if __name__ == '__main__':
    unittest.main()
