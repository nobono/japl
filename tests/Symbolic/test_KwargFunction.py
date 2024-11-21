import unittest
from sympy import Symbol
from japl.Symbolic.KwargFunction import KwargFunction
from sympy import pycode
from sympy import ccode
from sympy import cse
import dill as pickle
import io



class func(KwargFunction):
    pass


class func2(KwargFunction):
    pass


class TestKwargFunction(unittest.TestCase):


    def setUp(self) -> None:
        pass


    def test_case1(self):
        """no args"""
        f = func()
        self.assertEqual(str(f), "func()")
        self.assertEqual(f.args, ())
        self.assertEqual(f.kwargs, {})
        self.assertEqual(f.name, "func")


    def test_case2(self):
        """number args"""
        f = func(a=1, b=2)
        self.assertEqual(str(f), "func(a=1, b=2)")
        self.assertEqual(f.args, (1, 2))
        self.assertEqual(f.kwargs, {'a': 1, 'b': 2})
        self.assertEqual(f.name, "func")


    def test_case3(self):
        """symbol args"""
        b = Symbol('b')
        f = func(a=1, b=b)
        self.assertEqual(str(f), "func(a=1, b=b)")
        self.assertEqual(f.args, (1, b))
        self.assertEqual(f.kwargs, {'a': 1, 'b': b})
        self.assertEqual(f.name, "func")


    def test_pycode_case1(self):
        """symbol args"""
        b = Symbol('b')
        f = func(a=1, b=b)
        self.assertEqual(pycode(f), "func(a=1, b=b)")
        self.assertEqual(f.args, (1, b))
        self.assertEqual(f.kwargs, {'a': 1, 'b': b})
        self.assertEqual(f.name, "func")


    def test_ccode_case1(self):
        """symbol args"""
        b = Symbol('b')
        f = func(a=1, b=b)
        self.assertEqual(ccode(f), "func(py::kw(\"a\"_a=1), py::kw(\"b\"_a=b))")
        self.assertEqual(f.args, (1, b))
        self.assertEqual(f.kwargs, {'a': 1, 'b': b})
        self.assertEqual(f.name, "func")


    def test_add_case1(self):
        """addition"""
        b = Symbol('b')
        f = func(a=1, b=b)
        expr = f + 1  # type:ignore
        self.assertEqual(str(expr), "func(a=1, b=b) + 1")
        self.assertEqual(expr.args, (1, f))
        self.assertEqual(expr.args[1].args, (1, b))
        self.assertEqual(expr.args[1].kwargs, {'a': 1, 'b': b})


    def test_pickle_case1(self):
        b = Symbol('b')
        f = func(a=1, b=b)
        file = io.BytesIO()
        pickle.dump(f, file)
        file.seek(0)
        loaded_data = pickle.load(file)
        self.assertEqual(loaded_data.name, "func")
        self.assertEqual(loaded_data.args, (1, b))
        self.assertEqual(loaded_data.kwargs, {'a': 1, 'b': b})


    def test_subs_case1(self):
        b = Symbol('b')
        expr = func(a=1, b=b) + 1  # type:ignore
        expr = expr.subs(b, 2)
        self.assertEqual(expr, func(a=1, b=2) + 1)  # type:ignore


    def test_xreplace_case1(self):
        b = Symbol('b')
        expr = func(a=1, b=b) + 1  # type:ignore
        expr = expr.xreplace({b: 2})
        self.assertEqual(expr, func(a=1, b=2) + 1)  # type:ignore


    def test_contains_case1(self):
        a = Symbol('a')
        b = Symbol('b')
        c = a + b  # type:ignore
        expr = func(a=c, b=c, c=c)
        self.assertTrue(a in expr)
        self.assertTrue(b in expr)
        self.assertTrue(c in expr)


    def test_cse_case1(self):
        a = Symbol('a')
        b = Symbol('b')
        c = a + b  # type:ignore
        expr = func(a=c, b=c)
        rep, simp_expr = cse(expr)
        x0 = Symbol("x0")
        self.assertEqual(rep[0], (x0, a + b))  # type:ignore
        self.assertEqual(simp_expr[0], func(a=x0, b=x0))  # type:ignore


    # def test_doit_case1(self):
    #     a = Symbol('a')
    #     b = Symbol('b')
    #     c = a + b  # type:ignore
    #     expr = func(a=c, b=c)


    def test_compares_case1(self):
        self.assertTrue(id(func()) != id(func2()))
        self.assertEqual(func().name, "func")
        self.assertEqual(func2().name, "func2")


    def test_diff_case1(self):
        b = Symbol("b")
        self.assertEqual(func(a=1, b=1).diff(), 0)
        # print(func(a=1, b=b).diff(b))
        # self.assertEqual(func(a=1, b=b).diff(b), 0)


    # def test_compares_case1(self):
    #     from sympy import Piecewise
    #     from sympy import Function
    #     from sympy import Matrix
    #     a = Symbol('a')
    #     b = Symbol('b')
    #     c = a + b  # type:ignore
    #     d = Symbol('d')
    #     func = KwargFunction('func')
    #     func = func(a=c, b=c)

    #     func2 = KwargFunction('func2')(j=2, k=3)
    #     print(func)
    #     print(func2)

    #     # mat = Matrix
    #     # func = Function('func')(c, c)

    #     # expr = Piecewise(
    #     #         (func, d > 1),
    #     #         (0, True)
    #     #         )
    #     # ret = expr.subs(b, 1)
    #     # print(expr)
    #     # print(ret)
    #     # from sympy import StrictGreaterThan, StrictLessThan
    #     # print(StrictGreaterThan(expr, 1))


if __name__ == '__main__':
    unittest.main()
