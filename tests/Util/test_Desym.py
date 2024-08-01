import unittest
from japl.Util.Desym import Desym
from sympy import symbols, Max, Min



class TestDesym(unittest.TestCase):

    def test_desym_case1(self):
        """simple setup"""
        x, y = symbols("x y")
        expr = x + y + 2
        desym = Desym((x, y), expr)
        self.assertEqual(desym.vars, (x, y))


    def test_desym_case2(self):
        """simple expression"""
        x, y = symbols("x y")
        expr = x + y + 2
        desym = Desym((x, y), expr)
        self.assertEqual(desym(1, 2), 5)


    def test_desym_case3(self):
        """different argument format"""
        x, y, z = symbols("x y z")
        expr = x + y + z + 2
        desym = Desym(((x, y), z), expr)
        self.assertEqual(desym([1, 2], 3), 8)


    def test_desym_custom_modeuls_case1(self):
        """test custom modules used with sympy lambdify"""
        x, y = symbols("x y")
        expr = Max(x, y)
        desym = Desym((x, y), expr)
        subs = {x: 1, y: 2}
        truth = expr.subs(subs)
        self.assertEqual(desym(1, 2), truth)


    def test_desym_custom_modeuls_case2(self):
        """test custom modules used with sympy lambdify"""
        x, y = symbols("x y")
        expr = Min(x, y)
        desym = Desym((x, y), expr)
        subs = {x: 1, y: 2}
        truth = expr.subs(subs)
        self.assertEqual(desym(1, 2), truth)


if __name__ == '__main__':
    unittest.main()
