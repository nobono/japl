import unittest
import numpy as np
from japl.DataTable.DataTable import DataTable
import linterp
import model
import atmosphere
import aerotable
import datatable



class TestCppDataTable(unittest.TestCase):


    def setUp(self) -> None:
        self.TOLERANCE_PLACES = 15
        self.data = np.array([[1, 2, 3],
                              [4, 5, 6],
                              [7, 8, 9]],
                             dtype=float)
        self.axes = {"alpha": np.array([0., 1, 2]),
                     "mach": np.array([0., 1, 2])}


    def test_datatable_members(self):
        table = datatable.DataTable(self.data, self.axes)
        self.assertListEqual(list(table.axes.keys()), ["alpha", "mach"])
        self.assertIsInstance(table.interp, linterp.Interp2d)
        self.assertTrue((table.interp._data == self.data).all())


    def test_datatable_call(self):
        """test call datable with dict"""
        table = datatable.DataTable(self.data, self.axes)
        ret = table(dict(alpha=1., mach=1.))
        self.assertListEqual(ret.tolist(), [5.])

        """test call datable with kwargs"""
        table = datatable.DataTable(self.data, self.axes)
        ret = table(alpha=1., mach=1.)
        self.assertListEqual(ret.tolist(), [5.])

        """test call datable with iterable"""
        table = datatable.DataTable(self.data, self.axes)
        ret = table([[1., 1.]])
        self.assertListEqual(ret.tolist(), [5.])

        """test call datatable with list of points"""
        table = datatable.DataTable(self.data, self.axes)
        ret = table([[0, 0], [1, 1], [2, 2]])
        self.assertListEqual(ret.tolist(), [1, 5, 9])


if __name__ == '__main__':
    unittest.main()
