import unittest
import numpy as np
from sympy import Matrix, MatrixSymbol, symbols
from japl.Math.Vec import vec_ang
from japl.Math.VecSymbolic import vec_ang_sym



class TestMathRotation(unittest.TestCase):


    def setUp(self):
        self.TOLERANCE_PLACES = 15


    def test_vec_ang(self):
        vec1 = np.array([1, 2, 3])
        vec2 = np.array([4, 3, 2])
        ret = vec_ang(vec1, vec2)
        truth = 0.653325805624606071
        # print("%.18f" % ret)
        self.assertTrue(ret == truth)


    def test_vec_ang_sym(self):
        vec1 = MatrixSymbol("vec1", 3, 1)
        vec2 = MatrixSymbol("vec2", 3, 1)
        ret = vec_ang_sym(vec1, vec2)
        subs = {vec1[0]: 1,
                vec1[1]: 2,
                vec1[2]: 3,
                vec2[0]: 4,
                vec2[1]: 3,
                vec2[2]: 2,
                }
        ret = ret.subs(subs)
        truth = 0.653325805624606071
        self.assertTrue(ret.evalf() == truth)  # type:ignore


    def get_rand_vec(self):
        return np.random.random(3)


    def run_vec_ang(self):
        vec1 = self.get_rand_vec()
        vec2 = self.get_rand_vec()
        return (vec1, vec2, vec_ang(vec1, vec2))


    def run_vec_ang_sym(self):
        vec1 = Matrix(symbols("vec1_x, vec1_y, vec1_z"))
        vec2 = Matrix(symbols("vec2_x, vec2_y, vec2_z"))
        return vec_ang_sym(vec1, vec2)


    def test_vec_ang_compare(self):
        sympy_precision = 36
        ret1 = []
        ret2 = []
        for _ in range(10):
            (vec1,
             vec2,
             vec_ang_ret) = self.run_vec_ang()
            vec_ang_sym = self.run_vec_ang_sym()
            subs = {"vec1_x": vec1[0],
                    "vec1_y": vec1[1],
                    "vec1_z": vec1[2],
                    "vec2_x": vec2[0],
                    "vec2_y": vec2[1],
                    "vec2_z": vec2[2],
                    }
            vec_ang_sym_ret = vec_ang_sym.subs(subs).n(sympy_precision)  # type:ignore
            ret1 += [vec_ang_ret]
            ret2 += [vec_ang_sym_ret]
            self.assertAlmostEqual(vec_ang_ret, float(vec_ang_sym_ret), places=self.TOLERANCE_PLACES)  # type:ignore


if __name__ == '__main__':
    unittest.main()
