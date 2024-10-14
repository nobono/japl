import unittest
from sympy import symbols, Matrix, Function, Symbol, MatrixSymbol
from sympy import cse
from japl.BuildTools.CCodeGenerator import CCodeGenerator



class TestCCodeGenerator(unittest.TestCase):


    def setUp(self) -> None:
        pass


    def test_case1(self):
        t = symbols("t")
        dt = symbols("dt")
        pos_x = Function("pos_x", real=True)(t)  # type:ignore
        pos_y = Function("pos_y", real=True)(t)  # type:ignore
        pos_z = Function("pos_z", real=True)(t)  # type:ignore
        vel_x = Function("vel_x", real=True)(t)  # type:ignore
        vel_y = Function("vel_y", real=True)(t)  # type:ignore
        vel_z = Function("vel_z", real=True)(t)  # type:ignore
        acc_x = Symbol("acc_x", real=True)  # type:ignore
        acc_y = Symbol("acc_y", real=True)  # type:ignore
        acc_z = Symbol("acc_z", real=True)  # type:ignore
        pos = Matrix([pos_x, pos_y, pos_z])
        vel = Matrix([vel_x, vel_y, vel_z])
        acc = Matrix([acc_x, acc_y, acc_z])

        pos_new = vel * dt + 0.5 * acc * dt**2
        replacements, expr_simple = cse(pos_new)

        gen = CCodeGenerator()
        ret = gen.write_subexpressions(replacements)
        print(ret)
        self.assertEqual(ret, "\tdouble x0 = 0.5*pow(dt, 2);\n")


if __name__ == '__main__':
    unittest.main()
