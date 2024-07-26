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
        model = Model().from_function(dt, state, input, func)
        self.assertTrue(model.vars == (state, input, dt))
        self.assertTrue(model.update_func == func)
        self.assertTrue((model([0, 0, 0, 0, 0, 0], [1, 0, 0], 0.01) == np.array([0, 0, 0, 1, 0, 0])).all())


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
        model = Model().from_statespace(dt, state, input, A, B)
        self.assertTrue(model.vars == (state, input, dt))
        self.assertTrue((model.A == A).all())
        self.assertTrue((model.B == B).all())
        self.assertTrue((model([0, 0, 0, 0, 0, 0], [1, 0, 0], 0.01) == np.array([0, 0, 0, 1, 0, 0])).all())


    def test_from_expression(self):
        state, input, dt, dynamics = self.setup()
        model = Model().from_expression(dt, state, input, dynamics)
        self.assertTrue(model.vars == (state, input, dt))
        self.assertTrue(model.expr == dynamics)
        self.assertTrue((model([0, 0, 0, 0, 0, 0], [1, 0, 0], 0.01) == np.array([0, 0, 0, 1, 0, 0])).all())


if __name__ == '__main__':
    unittest.main()
