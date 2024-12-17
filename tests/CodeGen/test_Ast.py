import unittest
from sympy import symbols
from sympy import Matrix
from sympy import MatrixSymbol
# from sympy import ccode
from japl.CodeGen import ccode
from japl.CodeGen import pycode
from sympy.codegen.ast import String
from japl.CodeGen.Ast import Tuple
from japl.CodeGen.Ast import Kwargs
from japl.CodeGen.Ast import Dict
from japl.CodeGen.Ast import PyType
from japl.CodeGen.Ast import CType
from japl.CodeGen.Ast import CTypes
from japl.CodeGen.Ast import CodeGenFunctionCall
from sympy.codegen.ast import Variable
from sympy.codegen.ast import String



class TestAst(unittest.TestCase):


    def setUp(self) -> None:
        pass


    def test_CType_case1(self):
        double = CType("double")
        self.assertEqual(double.name, String("double"))
        self.assertEqual(ccode(double.as_vector()), "vector<double>")
        self.assertEqual(ccode(double.as_ndarray()), "py::array_t<double>")
        self.assertEqual(ccode(double.as_map()), "map<string, double>")
        self.assertEqual(ccode(double.as_const()), "const double")
        self.assertEqual(ccode(double.as_vector().as_const()), "const vector<double>")
        self.assertEqual(ccode(double.as_ref()), "double&")
        self.assertEqual(ccode(double.as_vector().as_const().as_ref()), "const vector<double>&")


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
        a = symbols("a")
        A = Matrix([1, 2, a])
        d = Kwargs(x=A)
        self.assertEqual(ccode(d), '{{"x", _Dummy_var0}}')


    def test_call_case_1(self):
        f = CodeGenFunctionCall("func", (1, 2), x=1, y=2)
        self.assertEqual(f.name, String("func"))
        self.assertEqual(f.function_args, Tuple(1, 2))
        self.assertEqual(f.function_kwargs, Kwargs(x=1, y=2))
        self.assertEqual(ccode(f.function_kwargs), '{{"x", 1}, {"y", 2}}')
        self.assertEqual(ccode(f), 'func(1, 2, {{"x", 1}, {"y", 2}})')
        self.assertEqual(pycode(f.function_kwargs), 'x=1, y=2')
        self.assertEqual(pycode(f), 'func(1, 2, x=1, y=2)')

        f = CodeGenFunctionCall("func", 1, 2, x=1, y=2)
        self.assertEqual(f.name, String("func"))
        self.assertEqual(f.function_args, Tuple(1, 2))
        self.assertEqual(f.function_kwargs, Kwargs(x=1, y=2))
        self.assertEqual(ccode(f.function_kwargs), '{{"x", 1}, {"y", 2}}')
        self.assertEqual(ccode(f), 'func(1, 2, {{"x", 1}, {"y", 2}})')
        self.assertEqual(pycode(f.function_kwargs), 'x=1, y=2')
        self.assertEqual(pycode(f), 'func(1, 2, x=1, y=2)')

        f = CodeGenFunctionCall("func", (1, 2), dict(x=1, y=2))
        self.assertEqual(f.name, String("func"))
        self.assertEqual(f.function_args, Tuple(1, 2))
        self.assertEqual(f.function_kwargs, Kwargs(x=1, y=2))
        self.assertEqual(ccode(f.function_kwargs), '{{"x", 1}, {"y", 2}}')
        self.assertEqual(ccode(f), 'func(1, 2, {{"x", 1}, {"y", 2}})')
        self.assertEqual(pycode(f.function_kwargs), 'x=1, y=2')
        self.assertEqual(pycode(f), 'func(1, 2, x=1, y=2)')


if __name__ == '__main__':
    unittest.main()
