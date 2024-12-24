import unittest
from sympy import symbols, Matrix, Symbol, MatrixSymbol
from japl import StateRegister



class TestStateRegister(unittest.TestCase):


    def setup(self):
        pass


    def test_set_Symbol(self):
        """collection of vars"""
        state = symbols("a, b, c, d")
        reg = StateRegister()
        reg.set(state)
        self.assertIn('a', reg)
        self.assertIn('b', reg)
        self.assertIn('c', reg)
        self.assertIn('d', reg)
        self.assertEqual(reg['a'], {"id": 0, "label": 'a', "var": symbols('a'), "size": 1})
        self.assertEqual(reg['b'], {"id": 1, "label": 'b', "var": symbols('b'), "size": 1})
        self.assertEqual(reg['c'], {"id": 2, "label": 'c', "var": symbols('c'), "size": 1})
        self.assertEqual(reg['d'], {"id": 3, "label": 'd', "var": symbols('d'), "size": 1})


    def test_set_Matrix(self):
        """vars and matrix"""
        mat = MatrixSymbol("m", 3, 3)
        state = [Symbol("a"), mat]
        reg = StateRegister()
        reg.set(state)
        self.assertIn('a', reg)
        self.assertIn('m', reg)
        self.assertEqual(reg['a'], {"id": 0, "label": 'a', "var": symbols('a'), "size": 1})
        self.assertEqual(reg['m'], {"id": 1, "label": 'm', "var": MatrixSymbol('m', 3, 3), "size": 9})


    def test_get_ids(self):
        state = symbols("a, b, c, d")
        reg = StateRegister()
        reg.set(state)
        self.assertEqual(reg.get_ids('a'), 0)
        self.assertEqual(reg.get_ids('b'), 1)
        self.assertEqual(reg.get_ids('c'), 2)
        self.assertEqual(reg.get_ids('d'), 3)

        """multiple"""
        self.assertListEqual(reg.get_ids(['a', 'b', 'c', 'd']), [0, 1, 2, 3])  # type:ignore

        """matrix"""
        m = MatrixSymbol("m", 3, 3)
        state = [Symbol("a"), m]
        reg = StateRegister()
        reg.set(state)
        m_ids = reg.get_ids("m")
        self.assertListEqual(m_ids, [[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # type:ignore
        self.assertEqual(reg.matrix_info, {'m': {'id': 1, 'label': 'm', 'var': m, 'size': 9}})


    def test_get_vars(self):
        state = symbols("a, b, c, d")
        reg = StateRegister()
        reg.set(state)
        self.assertEqual(reg.get_vars().flat(), list(state))

        """matrix"""
        mat = MatrixSymbol("m", 3, 3)
        state = [Symbol("a"), mat]
        reg = StateRegister()
        reg.set(state)
        self.assertListEqual(reg.get_vars().flat(), [Symbol("a")])  # NOTE: get_vars ignores MatrixSymbols




if __name__ == '__main__':
    unittest.main()
