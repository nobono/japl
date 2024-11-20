import unittest
from sympy import Symbol
from japl.Symbolic.KwargFunction import KwargFunction
from sympy import pycode
from sympy import ccode



class TestKwargFunction(unittest.TestCase):


    def setUp(self) -> None:
        pass


    def test_case1(self):
        """no args"""
        func = KwargFunction('func')
        self.assertEqual(str(func), "func()")
        self.assertEqual(func.args, ())
        self.assertEqual(func.kwargs, {})


    def test_case2(self):
        """number args"""
        func = KwargFunction('func')
        func = func(a=1, b=2)
        self.assertEqual(str(func), "func(a=1,b=2)")
        self.assertEqual(func.args, (1, 2))
        self.assertEqual(func.kwargs, {'a': 1, 'b': 2})


    def test_case3(self):
        """symbol args"""
        b = Symbol('b')
        func = KwargFunction('func')
        func = func(a=1, b=b)
        self.assertEqual(str(func), "func(a=1,b=b)")
        self.assertEqual(func.args, (1, b))
        self.assertEqual(func.kwargs, {'a': 1, 'b': b})


    def test_pycode_case1(self):
        """symbol args"""
        b = Symbol('b')
        func = KwargFunction('func')
        func = func(a=1, b=b)
        self.assertEqual(pycode(func), "func(a=1,b=b)")
        self.assertEqual(func.args, (1, b))
        self.assertEqual(func.kwargs, {'a': 1, 'b': b})


    def test_ccode_case1(self):
        """symbol args"""
        b = Symbol('b')
        func = KwargFunction('func')
        func = func(a=1, b=b)
        self.assertEqual(ccode(func), r"func(a=1,b=b)")
        self.assertEqual(func.args, (1, b))
        self.assertEqual(func.kwargs, {'a': 1, 'b': b})


if __name__ == '__main__':
    unittest.main()
