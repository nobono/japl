import unittest
import numpy as np
from sympy import Matrix, MatrixSymbol
from japl.Math.Vec import vec_ang
from japl.Math.VecSymbolic import vec_ang_sym



class TestMathRotation(unittest.TestCase):


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
        self.assertTrue(ret.evalf() == truth) #type:ignore


if __name__ == '__main__':
    unittest.main()
