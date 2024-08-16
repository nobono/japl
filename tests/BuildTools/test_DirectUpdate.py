import unittest
from sympy import symbols, Matrix, Function
from japl.BuildTools.DirectUpdate import DirectUpdate
from japl.BuildTools.DirectUpdate import DirectUpdateSymbol



class TestDirectUpdate(unittest.TestCase):


    def setUp(self) -> None:
        pass


    def test_DirectUpdateSymbol_case1(self):
        x, y = symbols("x y")
        ret = DirectUpdateSymbol("var", state_expr=x, sub_expr=y)
        self.assertEqual(ret.name, 'var')
        self.assertEqual(ret.state_expr, x)
        self.assertEqual(ret.sub_expr, y)


    def test_DirectUpdate_case1(self):
        """Symbols"""
        a, b, c = symbols("a b c")
        x, y, z = symbols("x y z")
        var = Matrix([a, b, c])
        val = Matrix([x, y, z])
        ret = DirectUpdate(var, val)
        self.assertIsInstance(ret, Matrix)
        self.assertIsInstance(ret[0], DirectUpdateSymbol)
        self.assertEqual(ret[0].state_expr, a)  # type:ignore
        self.assertEqual(ret[1].state_expr, b)  # type:ignore
        self.assertEqual(ret[2].state_expr, c)  # type:ignore
        self.assertEqual(ret[0].sub_expr, x)  # type:ignore
        self.assertEqual(ret[1].sub_expr, y)  # type:ignore
        self.assertEqual(ret[2].sub_expr, z)  # type:ignore


    def test_DirectUpdate_case2(self):
        """Matrices"""
        a, b, c = symbols("a b c")
        x, y, z = symbols("x y z")
        var = Matrix([a, b, c])
        val = Matrix([x + 1, 2 / y, 3 * z])
        ret = DirectUpdate(var, val)
        self.assertIsInstance(ret, Matrix)
        self.assertIsInstance(ret[0], DirectUpdateSymbol)
        self.assertEqual(ret[0].state_expr, a)  # type:ignore
        self.assertEqual(ret[1].state_expr, b)  # type:ignore
        self.assertEqual(ret[2].state_expr, c)  # type:ignore
        self.assertEqual(ret[0].sub_expr, x + 1)  # type:ignore
        self.assertEqual(ret[1].sub_expr, 2 / y)  # type:ignore
        self.assertEqual(ret[2].sub_expr, 3 * z)  # type:ignore


    def test_DirectUpdate_case3(self):
        """Functions"""
        t = symbols("t")
        a, b, c = symbols("a b c", cls=Function)  # type:ignore
        x, y, z = symbols("x y z", cls=Function)  # type:ignore
        var = Matrix([a(t), b(t), c(t)])
        val = Matrix([x(t), y(t), z(t)])
        ret = DirectUpdate(var, val)
        self.assertIsInstance(ret, Matrix)
        self.assertIsInstance(ret[0], DirectUpdateSymbol)
        self.assertEqual(ret[0].state_expr, a(t))  # type:ignore
        self.assertEqual(ret[1].state_expr, b(t))  # type:ignore
        self.assertEqual(ret[2].state_expr, c(t))  # type:ignore
        self.assertEqual(ret[0].sub_expr, x(t))  # type:ignore
        self.assertEqual(ret[1].sub_expr, y(t))  # type:ignore
        self.assertEqual(ret[2].sub_expr, z(t))  # type:ignore


    def test_DirectUpdate_case4(self):
        """lists"""
        a, b, c = symbols("a b c")
        x, y, z = symbols("x y z")
        var = [a, b, c]
        val = [x + 1, 2 / y, 3 * z]
        ret = DirectUpdate(var, val)
        self.assertIsInstance(ret, Matrix)
        self.assertIsInstance(ret[0], DirectUpdateSymbol)
        self.assertEqual(ret[0].state_expr, a)  # type:ignore
        self.assertEqual(ret[1].state_expr, b)  # type:ignore
        self.assertEqual(ret[2].state_expr, c)  # type:ignore
        self.assertEqual(ret[0].sub_expr, x + 1)  # type:ignore
        self.assertEqual(ret[1].sub_expr, 2 / y)  # type:ignore
        self.assertEqual(ret[2].sub_expr, 3 * z)  # type:ignore


if __name__ == '__main__':
    unittest.main()
