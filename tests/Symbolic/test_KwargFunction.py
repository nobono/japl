import unittest
from sympy import Symbol
from japl.Symbolic.KwargFunction import KwargFunction
from sympy import pycode
from sympy import ccode
import dill as pickle
import io



class TestKwargFunction(unittest.TestCase):


    def setUp(self) -> None:
        pass


    def test_case1(self):
        """no args"""
        func = KwargFunction('func')
        self.assertEqual(str(func), "func()")
        self.assertEqual(func.args, ())
        self.assertEqual(func.kwargs, {})
        self.assertEqual(func.name, "func")


    def test_case2(self):
        """number args"""
        func = KwargFunction('func')
        func = func(a=1, b=2)
        self.assertEqual(str(func), "func(a=1, b=2)")
        self.assertEqual(func.args, (1, 2))
        self.assertEqual(func.kwargs, {'a': 1, 'b': 2})
        self.assertEqual(func.name, "func")


    def test_case3(self):
        """symbol args"""
        b = Symbol('b')
        func = KwargFunction('func')
        func = func(a=1, b=b)
        self.assertEqual(str(func), "func(a=1, b=b)")
        self.assertEqual(func.args, (1, b))
        self.assertEqual(func.kwargs, {'a': 1, 'b': b})
        self.assertEqual(func.name, "func")


    def test_pycode_case1(self):
        """symbol args"""
        b = Symbol('b')
        func = KwargFunction('func')
        func = func(a=1, b=b)
        self.assertEqual(pycode(func), "func(a=1, b=b)")
        self.assertEqual(func.args, (1, b))
        self.assertEqual(func.kwargs, {'a': 1, 'b': b})
        self.assertEqual(func.name, "func")


    def test_ccode_case1(self):
        """symbol args"""
        b = Symbol('b')
        func = KwargFunction('func')
        func = func(a=1, b=b)
        self.assertEqual(ccode(func), "func(py::kw(\"a\"_a=1), py::kw(\"b\"_a=b))")
        self.assertEqual(func.args, (1, b))
        self.assertEqual(func.kwargs, {'a': 1, 'b': b})
        self.assertEqual(func.name, "func")


    def test_add_case1(self):
        """addition"""
        b = Symbol('b')
        func = KwargFunction('func')
        func = func(a=1, b=b)
        ret = func + 1
        self.assertEqual(str(ret), "func(a=1, b=b) + 1")
        self.assertEqual(ret.args, (1, func))
        self.assertEqual(ret.args[1].args, (1, b))
        self.assertEqual(ret.args[1].kwargs, {'a': 1, 'b': b})


    def test_pickle_case1(self):
        """symbol args"""
        b = Symbol('b')
        func = KwargFunction('func')
        func = func(a=1, b=b)
        file = io.BytesIO()
        pickle.dump(func, file)
        file.seek(0)
        loaded_data = pickle.load(file)
        self.assertEqual(loaded_data.name, "func")
        self.assertEqual(loaded_data.args, (1, b))
        self.assertEqual(loaded_data.kwargs, {'a': 1, 'b': b})


if __name__ == '__main__':
    unittest.main()
