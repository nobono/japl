import unittest
from textwrap import dedent
from sympy import symbols
from sympy import symbols
from japl.CodeGen.JaplFunction import JaplFunction
from japl.CodeGen.Ast import Constructor
from japl.CodeGen.Ast import CType, CTypes
from japl.CodeGen.Ast import PyType, PyTypes
from japl.CodeGen.Ast import Kwargs
from japl.CodeGen.Ast import Dict
from japl.CodeGen import ccode
from japl.CodeGen import pycode

from sympy.codegen.ast import CodeBlock
from sympy.codegen.ast import Assignment
from sympy.codegen.ast import FunctionPrototype
from sympy.codegen.ast import Return
from sympy.codegen.ast import Variable
from sympy.codegen.ast import Type
from sympy.codegen.ast import String
from sympy.codegen.ast import Token
from sympy.codegen.ast import Tuple
from sympy import Matrix, Symbol, MatrixSymbol




class func(JaplFunction):
    pass


class func2(JaplFunction):
    pass


class TestJaplFunction_CodeGen(unittest.TestCase):


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
        """MatrixSymbols"""
        ret = CTypes.from_expr(MatrixSymbol("A", 3, 1))
        self.assertEqual(str(ret), "vector<double>")
        with self.assertRaises(Exception):
            ret = CTypes.from_expr(MatrixSymbol("A", 3, 2))
        with self.assertRaises(Exception):
            ret = CTypes.from_expr(MatrixSymbol("A", 2, 3))


    def test_codegen_call_case1(self):
        a = symbols('a')
        f = func(a=1)
        self.assertEqual(f.name, "func")
        self.assertEqual(ccode(f), "func({{\"a\", 1}})")


    def test_codegen_call_case2(self):
        class method(JaplFunction):
            pass
        f = method(a=1)
        f.set_parent("obj")
        self.assertEqual(f.name, "obj.method")
        self.assertEqual(pycode(f), "obj.method(a=1)")
        self.assertEqual(ccode(f), "obj.method({{\"a\", 1}})")


    def test_codegen_call_case3(self):
        class method(JaplFunction):
            parent = "obj"
            pass
        f = method(a=1)
        self.assertEqual(f.name, "obj.method")
        self.assertEqual(pycode(f), "obj.method(a=1)")
        self.assertEqual(ccode(f), "obj.method({{\"a\", 1}})")


    def test_codegen_call_case4(self):
        """dealing with Matrices as parameters"""
        a, b = symbols("a, b")
        c, d = symbols("c, d")
        A = Matrix([b])
        B = Matrix([c, d])
        f = func(a, A, B)
        self.assertEqual(ccode(f), "func(a, _Dummy_var0, _Dummy_var1)")
        self.assertEqual(pycode(f), "func(a, _Dummy_var0, _Dummy_var1)")


    def test_get_parameter_variables(self):
        code_type = 'c'
        a, b = symbols("a, b")
        f = func(a, b=1, c=2)
        ret = f._get_parameter_variables(code_type=code_type)
        self.assertEqual(ret[0].symbol.name, 'a')
        self.assertEqual(ret[1].symbol.name, '_Dummy_var0')
        self.assertEqual(ret[0].type, CTypes.float64.as_ref())
        self.assertEqual(ret[1].type, CTypes.float64.as_map().as_ref())
        code_type = 'py'
        ret = f._get_parameter_variables(code_type=code_type)
        self.assertEqual(ret[1].symbol.name, '_Dummy_var0')
        self.assertEqual(ret[0].type, PyTypes.float64.as_ref())
        self.assertEqual(ret[1].type, PyTypes.float64.as_map().as_ref())


    def test_codegen_build_proto_ccode(self):
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
        self.assertEqual(ccode(f.codegen_function_proto), "vector<double> func(map<string, double>& _Dummy_var0)")
        f = func(a, b=1)
        f._build_proto(expr=Matrix([c + d]), code_type=code_type)
        self.assertEqual(ccode(f.codegen_function_proto),
                         "vector<double> func(double& a, map<string, double>& _Dummy_var0)")


    def test_codegen_build_proto_dummy_params(self):
        code_type = 'c'
        a, b = symbols("a, b")
        f = func(1, a, 2.0)
        f._build_proto(expr=None, code_type=code_type)
        proto = f.get_proto()
        self.assertEqual(ccode(proto), "void func(double& _Dummy_var0, double& a, double& _Dummy_var1)")


    def test_codegen_build_def_case1(self):
        code_type = 'c'
        a, b = symbols("a, b")
        c, d = symbols("c, d")
        f = func(a, b)
        f._build_proto(expr=None, code_type=code_type)
        f._build_def(expr=None, code_type=code_type)
        truth = """\
                void func(double& a, double& b){

                }"""
        self.assertEqual(ccode(f.codegen_function_def), dedent(truth))


    def test_to_codeblock(self):
        a, b, c, d = symbols("a, b, c, d")
        ret = JaplFunction._to_codeblock(a)
        self.assertEqual(ccode(ret), "double a;")
        ret = JaplFunction._to_codeblock(a + b)
        truth = """\
                double _Ret_arg;
                _Ret_arg = a + b;
                return _Ret_arg;"""
        self.assertEqual(ccode(ret), dedent(truth))


    def test_to_codeblock_matrix(self):
        a, b, c, d = symbols("a, b, c, d")
        A = Matrix([a, b, c])
        ret = JaplFunction._to_codeblock(A)
        truth = """\
                vector<double> _Ret_arg;
                _Ret_arg[0] = a;
                _Ret_arg[1] = b;
                _Ret_arg[2] = c;
                return _Ret_arg;"""
        self.assertEqual(ccode(ret), dedent(truth))

        A = MatrixSymbol("A", 3, 1)
        ret = JaplFunction._to_codeblock(A * 2)
        truth = """\
                vector<double> _Ret_arg;
                _Ret_arg[0] = 2*A[0];
                _Ret_arg[1] = 2*A[1];
                _Ret_arg[2] = 2*A[2];
                return _Ret_arg;"""
        self.assertEqual(ccode(ret), dedent(truth))


    def test_to_codeblock_iterable(self):
        a, b, c, d = symbols("a, b, c, d")
        ret = JaplFunction._to_codeblock([
            a, b, c, d,
            Assignment(a, b),
            Assignment(c, d),
            Return(a)
            ])
        truth = """\
                double a;
                double b;
                double c;
                double d;
                a = b;
                c = d;
                return a;"""
        self.assertEqual(ccode(ret), dedent(truth))



    def test_to_constructor(self):
        var = Variable("a", type=CTypes.float64).as_Declaration()
        self.assertEqual(ccode(Constructor(var)), "double a()")
        var = Variable("a", type=CTypes.float64).as_Declaration()
        params = (1, 2)
        self.assertEqual(ccode(Constructor(var, params)), "double a(1, 2)")
        var = Variable("a", type=CTypes.float64)
        params = (1, 2)
        self.assertEqual(ccode(Constructor(var, params)), "double a(1, 2)")
        var = Variable("a", type=CTypes.float64)
        params = (Dict(dict({'a': 1, 'b': 2})),)
        self.assertEqual(ccode(Constructor(var, params)), 'double a({{"a", 1}, {"b", 2}})')


    # def test_(self):
    #     a, b = symbols("a, b")
    #     class fun(JaplFunction):  # type:ignore
    #         expr = a + b
    #     f = fun(1, 2)
    #     self.assertEqual(f.expr, a + b)
    #     f._build_function(code_type='c')
    #     print(ccode(f.get_proto()))


    # def test_codegen_build_function(self):
    #     code_type = 'c'
    #     a, b = symbols("a, b")
    #     c, d = symbols("c, d")
    #     f = func(a, b)
        # ret_var = Variable("ret", type=CTypes.float64).as_Declaration()
        # f._build_proto(expr=Symbol("ret"), code_type=code_type)
        # f._build_def(expr=[ret_var], code_type=code_type)
        # # truth = """\
        # #         void func(double& a, double& b){

        # #         }"""
        # # self.assertEqual(ccode(f.codegen_function_def), dedent(truth))
        # print(ccode(f.codegen_function_def))


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
