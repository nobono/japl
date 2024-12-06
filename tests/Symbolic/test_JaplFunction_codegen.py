import unittest
from sympy import symbols
from japl.Symbolic.JaplFunction import JaplFunction
from sympy import pycode
from sympy import ccode
from sympy import symbols



class func(JaplFunction):
    pass


class func2(JaplFunction):
    pass


class TestJaplFunction(unittest.TestCase):


    def setUp(self) -> None:
        pass

    # -----------------------------------------
    # Func definition codegen
    # -----------------------------------------

    def test_call_case_1(self):
        self.assertEqual(func().args, ())


    def test_call_case_2(self):
        f = func(1, 2)
        self.assertEqual(f.args, (1, 2))
        self.assertEqual(f.fargs, (1, 2))


    def test_call_case_3(self):
        f = func(1, 2, x=3)
        self.assertEqual(f.args, (1, 2, 3))
        self.assertEqual(f.fargs, (1, 2))
        self.assertEqual(f.kwargs, {'x': 3})


    # def test_params_case_1(self):
    #     f = func(1, 2, x=3)
    #     print(f.codegen_function_call.args)


    # def test_proto_case_1(self):
    #     self.assertEqual(func().args, ())


    # def test_def_case_1(self):
    #     self.assertEqual(func().args, ())


    # def test_def_build_case_2(self):
    #     a, b = symbols("a, b")
    #     f = func(a, b)
    #     print(ccode(f))
    #     self.assertEqual(f.args, (a, b))


if __name__ == '__main__':
    unittest.main()
