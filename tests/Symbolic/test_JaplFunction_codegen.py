import unittest
from sympy import symbols
from sympy import pycode
# from sympy import ccode
from sympy import symbols
from japl.Symbolic.JaplFunction import JaplFunction
from japl.Symbolic.JaplFunction import CodeGenUtil
from japl.Symbolic.Ast import CodeGenFunctionPrototype
from japl.Symbolic.Ast import CType, CTypes
from japl.Symbolic.Ast import Kwargs
from japl.Symbolic.Ast import Dict
# from japl.Symbolic.Ast import CodeGenPrinter
from japl.Symbolic.Ast import ccode

from sympy.codegen.ast import CodeBlock, Assignment, FunctionPrototype, Variable, Type, String, Token
from sympy import Matrix, Symbol, MatrixSymbol




class func(JaplFunction):
    pass


class func2(JaplFunction):
    pass


class TestJaplFunction(unittest.TestCase):


    def setUp(self) -> None:
        pass

    # -----------------------------------------
    # Codegen Util
    # -----------------------------------------

    def test_codegen_from_expr(self):
        ret = CTypes.from_expr(Symbol("a"))
        self.assertEqual(str(ret), "double")
        ret = CTypes.from_expr(Symbol("a", real=True))
        self.assertEqual(str(ret), "double")
        ret = CTypes.from_expr(Symbol("a", integer=True))
        self.assertEqual(str(ret), "int")
        ret = CTypes.from_expr(Symbol("a", boolean=True))
        self.assertEqual(str(ret), "bool")
        ret = CTypes.from_expr(Matrix([Symbol("a")]))
        self.assertEqual(str(ret), "vector<double>")
        ret = CTypes.from_expr(Matrix([Symbol("a", real=True)]))
        self.assertEqual(str(ret), "vector<double>")
        ret = CTypes.from_expr(Kwargs(x=1))
        self.assertEqual(str(ret), "map<string, double>")
        ret = CTypes.from_expr(Dict({'x': 1}))
        self.assertEqual(str(ret), "map<string, double>")


    def test_get_parameter_variables(self):
        code_type = 'c'
        a, b = symbols("a, b")
        f = func(a, b=1)
        ret = f._get_parameter_variables(code_type=code_type)
        self.assertEqual(ccode(ret[0]), 'a')
        self.assertEqual(ccode(ret[1]), 'kwargs')


    def test_codegen_build_proto(self):
        code_type = 'c'
        a, b = symbols("a, b")
        c, d = symbols("c, d")
        f = func(a, b)
        f._build_proto(expr=None, code_type=code_type)
        self.assertEqual(ccode(f.codegen_function_proto), "void func(double& a, double& b)")
        f = func(symbols("a, b", integer=True))
        f._build_proto(expr=c + d, code_type=code_type)
        self.assertEqual(ccode(f.codegen_function_proto), "double func(int& a, int& b)")
        f = func(a, b)
        f._build_proto(expr=Matrix([c + d]), code_type=code_type)
        self.assertEqual(ccode(f.codegen_function_proto), "vector<double> func(double& a, double& b)")
        f = func(b=1)
        f._build_proto(expr=Matrix([c + d]), code_type=code_type)
        self.assertEqual(ccode(f.codegen_function_proto), "vector<double> func(map<string, double>& kwargs)")
        f = func(a, b=1)
        f._build_proto(expr=Matrix([c + d]), code_type=code_type)
        self.assertEqual(ccode(f.codegen_function_proto), "vector<double> func(double& a, map<string, double>& kwargs)")


    # -----------------------------------------
    # Func definition codegen
    # -----------------------------------------

    # def test_call_case_1(self):
    #     self.assertEqual(func().args, ())


    # def test_call_case_2(self):
    #     f = func(1, 2)
    #     self.assertEqual(f.args, (1, 2))
    #     self.assertEqual(f.fargs, (1, 2))


    # def test_call_case_3(self):
    #     f = func(1, 2, x=3)
    #     self.assertEqual(f.args, (1, 2, 3))
    #     self.assertEqual(f.fargs, (1, 2))
    #     self.assertEqual(f.kwargs, {'x': 3})


if __name__ == '__main__':
    unittest.main()
