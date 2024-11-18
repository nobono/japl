import unittest
import numpy as np
from japl.DataTable.DataTable import DataTable



class TestDataTable(unittest.TestCase):


    def setUp(self) -> None:
        self.axes = {"alpha": np.array([0., 1, 2]),
                     "mach": np.array([0., 1, 2])}
        self.data = np.array([[0, 1, 2],
                              [3, 4, 5],
                              [6, 7, 8]], dtype=float)


    def test_case1(self):
        table = DataTable(self.data, self.axes)
        axis_keys = list(table.axes.keys())
        self.assertTrue((table == self.data).all())
        self.assertListEqual(axis_keys, ["alpha", "mach"])
        self.assertListEqual(table.axes["alpha"].tolist(), self.axes["alpha"].tolist())
        self.assertListEqual(table.axes["mach"].tolist(), self.axes["mach"].tolist())


    def test_required_args(self):
        table = DataTable(self.data, self.axes)
        with self.assertRaises(Exception):
            table(alpha=1)


    def test_mirror_axis_case1(self):
        table = DataTable(self.data, self.axes)
        true = np.array([[-6., -7., -8.],
                         [-3., -4., -5.],
                         [0., 1., 2.],
                         [3., 4., 5.],
                         [6., 7., 8.]])
        mtable = table.mirror_axis("alpha")
        self.assertTrue((mtable == true).all())
        self.assertListEqual(mtable.axes["alpha"].tolist(), [-2, -1, 0, 1, 2])
        self.assertListEqual(mtable.axes["mach"].tolist(), self.axes["mach"].tolist())


    def test_mirror_axis_case2(self):
        table = DataTable(self.data, self.axes)
        true = np.array([[-2., -1., 0., 1., 2.],
                         [-5., -4., 3., 4., 5.],
                         [-8., -7., 6., 7., 8.]])
        mtable = table.mirror_axis("mach")
        self.assertTrue((mtable == true).all())
        self.assertListEqual(mtable.axes["mach"].tolist(), [-2, -1, 0, 1, 2])
        self.assertListEqual(mtable.axes["alpha"].tolist(), self.axes["alpha"].tolist())



if __name__ == '__main__':
    unittest.main()
