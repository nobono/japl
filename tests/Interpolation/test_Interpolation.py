import unittest
import numpy as np
from japl.Interpolation.Interpolation import LinearInterp
from scipy.interpolate import RegularGridInterpolator



class TestInterpolation(unittest.TestCase):


    def setUp(self) -> None:
        self.TOLERANCE_PLACES = 15
        pass


    def test_case1(self):
        data = np.array([[1., 2, 3],
                         [-10., -20, -30]])
        daxes = {"alpha": np.array([0., 1]),
                 "mach": np.array([0., 5, 10])}
        axes = tuple([*daxes.values()])
        interp = LinearInterp(axes, data)
        rgi = RegularGridInterpolator(axes, data)
        args = [[0, 1], [0, 2]]
        true = rgi(args).tolist()
        ret = interp(args)
        self.assertAlmostEqual(true[0], ret[0], places=self.TOLERANCE_PLACES)
        self.assertAlmostEqual(true[1], ret[1], places=self.TOLERANCE_PLACES)

        args = np.array([[0, 1], [0, 2]])
        true = rgi(args).tolist()
        ret = interp(args).tolist()
        self.assertAlmostEqual(true[0], ret[0], places=self.TOLERANCE_PLACES)
        self.assertAlmostEqual(true[1], ret[1], places=self.TOLERANCE_PLACES)



if __name__ == '__main__':
    unittest.main()
