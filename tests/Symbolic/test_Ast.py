import unittest
from sympy import symbols
from sympy import pycode
from sympy import ccode
from sympy import symbols
from sympy.codegen.ast import String
from japl.Symbolic.Ast import Tuple
from japl.Symbolic.Ast import Kwargs
from japl.Symbolic.Ast import Dict
from japl.Symbolic.Ast import CTypes
from japl.Symbolic.Ast import CodegenFunctionCall
from sympy.codegen.ast import Variable



class TestAst(unittest.TestCase):


    def setUp(self) -> None:
        pass


    def test_dict_case1(self):
        self.assertEqual(Dict({'x': 1, 'y': 2}).kwpairs, {'x': 1, 'y': 2})
        self.assertEqual(Dict(dict(x=1, y=2)).kwpairs, {'x': 1, 'y': 2})
        d = Dict(dict(x=1, y=2))
        self.assertEqual(ccode(d), '{{"x", 1}, {"y", 2}}')
        self.assertEqual(pycode(d), '{x: 1, y: 2}')


    def test_kwargs_case1(self):
        self.assertEqual(Kwargs({'x': 1, 'y': 2}).kwpairs, {'x': 1, 'y': 2})
        self.assertEqual(Kwargs(x=1, y=2).kwpairs, {'x': 1, 'y': 2})
        d = Kwargs(x=1, y=2)
        self.assertEqual(ccode(d), '{{"x", 1}, {"y", 2}}')
        self.assertEqual(pycode(d), 'x=1, y=2')


    def test_call_case_1(self):
        f = CodegenFunctionCall("func", (1, 2), x=1, y=2)
        self.assertEqual(f.name, String("func"))
        self.assertEqual(f.function_args, Tuple(1, 2))
        self.assertEqual(f.function_kwargs, Kwargs(x=1, y=2))
        self.assertEqual(ccode(f.function_kwargs), '{{"x", 1}, {"y", 2}}')
        self.assertEqual(ccode(f), 'func(1, 2, {{"x", 1}, {"y", 2}})')
        self.assertEqual(pycode(f.function_kwargs), 'x=1, y=2')
        self.assertEqual(pycode(f), 'func(1, 2, x=1, y=2)')

        f = CodegenFunctionCall("func", (1, 2), dict(x=1, y=2))
        self.assertEqual(f.name, String("func"))
        self.assertEqual(f.function_args, Tuple(1, 2))
        self.assertEqual(f.function_kwargs, Kwargs(x=1, y=2))
        self.assertEqual(ccode(f.function_kwargs), '{{"x", 1}, {"y", 2}}')
        self.assertEqual(ccode(f), 'func(1, 2, {{"x", 1}, {"y", 2}})')
        self.assertEqual(pycode(f.function_kwargs), 'x=1, y=2')
        self.assertEqual(pycode(f), 'func(1, 2, x=1, y=2)')


if __name__ == '__main__':
    unittest.main()
