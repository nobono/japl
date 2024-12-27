import unittest
import numpy as np
from sympy import symbols, Matrix, Symbol, MatrixSymbol
from japl.Model.Model import Model
from japl.Model.StateRegister import StateRegister
from japl.SimObject.SimObject import SimObject



class TestSimObject_Integration(unittest.TestCase):


    def setup(self):
        pass


    def test_case1(self):
        dt = Symbol("dt")
        mat = MatrixSymbol("m", 3, 3)
        a = Symbol("a")
        state = [a, *mat]
        model = Model.from_expression(dt_var=dt,
                                      state_vars=state,
                                      input_vars=[],
                                      dynamics_expr=Matrix([]),
                                      use_multiprocess_build=False)

        simobj = SimObject(model)
        a0 = 0
        m0 = np.array([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]])
        simobj.init_state([a0, m0])
        m = simobj.get_state_array(simobj.X0, "m")
        m00 = simobj.get_state_array(simobj.X0, "m[0, 0]")

        self.assertEqual(m00, 1)
        self.assertEqual(m.shape, (3, 3))
        self.assertEqual(m[0, 0], 1)
        self.assertEqual(m[0, 1], 2)
        self.assertEqual(m[0, 2], 3)
        self.assertEqual(m[1, 0], 4)
        self.assertEqual(m[1, 1], 5)
        self.assertEqual(m[1, 2], 6)
        self.assertEqual(m[2, 0], 7)
        self.assertEqual(m[2, 1], 8)
        self.assertEqual(m[2, 2], 9)


if __name__ == '__main__':
    unittest.main()
