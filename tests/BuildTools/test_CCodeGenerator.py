import os
import subprocess
import shutil
import unittest
import sympy as sp
from textwrap import dedent
from sympy import symbols, Matrix, Symbol, MatrixSymbol
from sympy import cse
from japl.BuildTools.CCodeGenerator import CCodeGenerator
from japl import JAPL_HOME_DIR




class TestCCodeGenerator(unittest.TestCase):


    def setUp(self) -> None:
        pass


    def get_model(self) -> tuple:
        dt = symbols("dt")
        pos = Matrix(symbols("pos_x, pos_y, pos_z"))
        vel = Matrix(symbols("vel_x, vel_y, vel_z"))
        acc = Matrix(symbols("acc_x, acc_y, acc_z"))
        expr = pos + vel * dt + 0.5 * acc * dt**2
        params = [pos, vel, acc, dt]
        return (expr, params)


    def get_small_model(self) -> tuple:
        a = Symbol("a")
        b = Symbol("b")
        expr = a + b  # type:ignore
        params = [a, b]
        return (expr, params)


    def test_indent_lines(self):
        string = "double a = 1;\ndouble b = 2;\n"
        ret = CCodeGenerator._indent_lines(string)
        self.assertEqual(ret, "\tdouble a = 1;\n\tdouble b = 2;\n")

    #########################################
    # Get Type
    #########################################

    def test_get_type_case1(self):
        a = Symbol("a")
        ret = CCodeGenerator._get_type(a)
        self.assertEqual(ret, "double")


    def test_get_type_case2(self):
        a = Symbol("a", integer=True)
        ret = CCodeGenerator._get_type(a)
        self.assertEqual(ret, "int")


    def test_get_type_case3(self):
        a = Symbol("a", boolean=True)
        ret = CCodeGenerator._get_type(a)
        self.assertEqual(ret, "bool")


    def test_get_type_case4(self):
        A = MatrixSymbol("A", 3, 1).as_mutable()
        ret = CCodeGenerator._get_type(A)
        self.assertEqual(ret, "std::vector<double>")

    #########################################
    # Declare Variables
    #########################################

    def test_declare_variable_case1(self):
        a = symbols("a")
        ret = CCodeGenerator._declare_variable(a)
        self.assertEqual(ret, "double a;\n")


    def test_declare_variable_case2(self):
        a = symbols("a", integer=True)
        ret = CCodeGenerator._declare_variable(a)
        self.assertEqual(ret, "int a;\n")


    def test_declare_variable_case3(self):
        A = Matrix(symbols("a, b, c"))
        A_str = CCodeGenerator._declare_variable(A, force_name="A")
        B = MatrixSymbol("A", 3, 3)
        B_str = CCodeGenerator._declare_variable(B, force_name="B")
        self.assertEqual(A_str, "std::vector<double> A(3);\n")
        self.assertEqual(B_str, "std::vector<double> B(9);\n")


    def test_declare_variable_case4(self):
        a, b = symbols("a, b")
        with self.assertRaises(Exception):
            CCodeGenerator._declare_variable(a * b)

    #########################################
    # Declare Variables
    #########################################

    def test_declare_parameter_case1(self):
        a = symbols("a")
        ret = CCodeGenerator._declare_parameter(a)
        self.assertEqual(ret, "double a")


    def test_declare_parameter_case2(self):
        a = symbols("a", integer=True)
        ret = CCodeGenerator._declare_parameter(a)
        self.assertEqual(ret, "int a")


    def test_declare_parameter_case3(self):
        A = Matrix(symbols("a, b, c"))
        A_str = CCodeGenerator._declare_parameter(A, force_name="A")
        B = MatrixSymbol("A", 3, 3)
        B_str = CCodeGenerator._declare_parameter(B, force_name="B")
        self.assertEqual(A_str, "std::vector<double> A")
        self.assertEqual(B_str, "std::vector<double> B")


    def test_declare_parameter_case4(self):
        a, b = symbols("a, b")
        with self.assertRaises(Exception):
            CCodeGenerator._declare_parameter(a * b)

    #########################################
    # Subexpression
    #########################################

    def test_function_parameters_case1(self):
        x, y = sp.symbols("x, y")
        params = [x, y]
        gen = CCodeGenerator()
        arg_names, arg_unpack_str = gen._get_function_parameters(params=params)
        self.assertEqual(arg_names, "double x, double y")
        self.assertEqual(arg_unpack_str, "")


    def test_function_parameters_case2(self):
        A = MatrixSymbol("A", 3, 1).as_mutable()
        x, y = sp.symbols("x, y")
        params = [x, y, A]
        gen = CCodeGenerator()
        arg_names, arg_unpack_str = gen._get_function_parameters(params=params)
        self.assertEqual(arg_names, "double x, double y, std::vector<double> A")
        self.assertEqual(arg_unpack_str, "")


    def test_function_parameters_case3(self):
        A = Matrix(symbols("a1, a2, a3"))
        x = Symbol("x")
        y = Symbol("y")
        params = [x, y, *A]
        gen = CCodeGenerator()
        arg_names, arg_unpack_str = gen._get_function_parameters(params=params)
        self.assertEqual(arg_names, "double x, double y, double a1, double a2, double a3")
        self.assertEqual(arg_unpack_str, "")
        # print(arg_names)
        # print(arg_unpack_str)

    # TODO: handle MatrixSymbol
    # def test_function_parameters_case4(self):
    #     A = MatrixSymbol("A", 3, 1)
    #     x = Symbol("x")
    #     y = Symbol("y")
    #     params = [x, y, *A]
    #     gen = CCodeGenerator()
    #     arg_names, arg_unpack_str = gen.get_function_parameters(params=params)
    #     # self.assertEqual(arg_names, "double x, double y, double a1, double a2, double a3")
    #     # self.assertEqual(arg_unpack_str, "")
    #     # print(arg_unpack_str)

    #########################################
    # Function Return
    #########################################

    def test_write_function_returns_case1(self):
        """single var"""
        expr, _ = self.get_small_model()
        gen = CCodeGenerator()
        ret = gen._write_function_returns(expr=expr, return_names=["Ret"])
        self.assertEqual(ret, "return Ret;\n")  # }


    def test_write_function_returns_case2(self):
        """array"""
        expr, _ = self.get_model()
        gen = CCodeGenerator()
        ret = gen._write_function_returns(expr=expr, return_names=["Ret"])
        self.assertEqual(ret, "return py::array_t<double>(Ret.size(), Ret.data());\n")

    # def test_write_function_returns_case3(self):
    #     """matrix"""
    #     A = MatrixSymbol("A", 3, 3).as_mutable()
    #     b = Symbol("b")
    #     expr = A * b
    #     gen = CCodeGenerator()
    #     ret = gen._write_function_returns(expr=expr, return_names=["Ret"])
    #     # self.assertEqual(ret, "return py::array_t<double>(Ret.size(), Ret.data());\n")
    #     print(expr)
    #     print(ret)

    #########################################
    # Instantiate return array
    #########################################

    def test_instantiate_return_variable_case1(self):
        """single var"""
        expr, _ = self.get_small_model()
        gen = CCodeGenerator()
        ret = gen._instantiate_return_variable(expr=expr, name="Ret")
        self.assertEqual(ret, "double Ret;\n")


    def test_instantiate_return_variable_case2(self):
        """array"""
        expr, _ = self.get_model()
        gen = CCodeGenerator()
        ret = gen._instantiate_return_variable(expr=expr, name="Ret")
        self.assertEqual(ret, "std::vector<double> Ret(3);\n")

    # def test_instantiate_return_variable_case3(self):
    #     """matrix"""
    #     A = MatrixSymbol("A", 3, 3).as_mutable()
    #     b = Symbol("b")
    #     expr = A * b
    #     gen = CCodeGenerator()
    #     ret = gen._instantiate_return_variable(expr=expr, name="Ret")
    #     # self.assertEqual(ret, "std::vector<double> Ret(3);\n")
    #     print(expr)
    #     print(ret)

    #########################################
    # Function Building
    #########################################

    def test_write_function_definition_case1(self):
        """single vars - type double"""
        expr, params = self.get_small_model()
        gen = CCodeGenerator()
        ret = gen._write_function_definition(name="func", expr=expr, params=params)
        self.assertEqual(ret, "double func(double a, double b) {\n\n")  # }


    def test_write_function_definition_case2(self):
        """single vars - type int"""
        a = Symbol("a", integer=True)
        b = Symbol("b", integer=True)
        expr = a + b  # type:ignore
        params = [a, b]
        gen = CCodeGenerator()
        ret = gen._write_function_definition(name="func", expr=expr, params=params)
        self.assertEqual(ret, "int func(int a, int b) {\n\n")  # }


    def test_write_function_definition_case3(self):
        """arrays and single vars"""
        expr, params = self.get_model()
        gen = CCodeGenerator()
        ret = gen._write_function_definition(name="func", expr=expr, params=params)
        truth = """\
        py::array_t<double> func(std::vector<double> _Dummy_var0, std::vector<double> _Dummy_var1, std::vector<double> _Dummy_var2, double dt) {
        \tdouble pos_x = _Dummy_var0[0];
        \tdouble pos_y = _Dummy_var0[1];
        \tdouble pos_z = _Dummy_var0[2];
        \tdouble vel_x = _Dummy_var1[0];
        \tdouble vel_y = _Dummy_var1[1];
        \tdouble vel_z = _Dummy_var1[2];
        \tdouble acc_x = _Dummy_var2[0];
        \tdouble acc_y = _Dummy_var2[1];
        \tdouble acc_z = _Dummy_var2[2];
        """ # noqa
        self.assertEqual(ret, dedent(truth))


    def test_write_subexpressions_case1(self):
        dt = symbols("dt")
        pos = Matrix(symbols("pos_x, pos_y, pos_z"))
        vel = Matrix(symbols("vel_x, vel_y, vel_z"))
        acc = Matrix(symbols("acc_x, acc_y, acc_z"))

        pos_new = pos + vel * dt + 0.5 * acc * dt**2
        replacements, _ = cse(pos_new)

        gen = CCodeGenerator()
        ret = gen._write_subexpressions(replacements)
        self.assertEqual(ret, "const double x0 = 0.5*pow(dt, 2);\n")


    def test_add_function(self):
        a = Symbol("a")
        b = Symbol("b")
        expr, params = self.get_small_model()
        function_name = "func"
        return_name = "ret"
        gen = CCodeGenerator()
        gen.add_function(expr=expr,
                         params=params,
                         function_name=function_name,
                         return_name=return_name)
        reg_func = gen.function_register[function_name]
        truth = {'expr': a + b,  # type:ignore
                 'params': [a, b],
                 'return_name': 'ret',
                 'use_cse': True,
                 'is_symmetric': False,
                 'description': ''}
        self.assertTrue(truth, reg_func)


    #########################################
    # Integration Test
    #########################################

    # def test_build_dev_model(self):
    #     FILE_DIR = os.path.dirname(__file__)
    #     ROOT_DIR = f"{FILE_DIR}/../.."
    #     DEV_DIR = f"{ROOT_DIR}/japl/Library/_dev"

    #     ext_model_dir = f"{DEV_DIR}/_dev_model"
    #     if os.path.isdir(ext_model_dir):
    #         shutil.rmtree(ext_model_dir, ignore_errors=True)

    #     # compile extension module src files
    #     res = subprocess.run(["python", f"{DEV_DIR}/DevModel.py"], check=True, capture_output=True, text=True)

    #     # build extension module
    #     res = subprocess.run(["python", f"{ext_model_dir}/build.py", "build_ext", "--inplace"], check=True, capture_output=True)

    #     # import extension module and run tests
    #     import _dev_model  # type:ignore
    #     from japl.Library._dev.DevModel import truth
    #     args = (1, 1, 2, 3, [1, 2, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9])
    #     ret = _dev_model.func(*args)
    #     self.assertListEqual(truth, ret.tolist())

    #     # os.system(f"{ext_model_dir}/build.py clean")
    #     subprocess.run(["python", f"{ext_model_dir}/build.py clean"], check=True)
    #     if os.path.isdir(ext_model_dir):
    #         shutil.rmtree(ext_model_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
