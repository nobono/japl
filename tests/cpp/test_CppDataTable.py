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


    # def test_aero_table(self):
    #     """test aerotable table usage"""
    #     table = datatable.DataTable(self.data, self.axes)
    #     table2 = datatable.DataTable(self.data * 2, self.axes)
    #     aero = aerotable.AeroTable(CA=table, CNB=table2)
    #     # self.assertListEqual(aero.CA(alpha=1, mach=1).tolist(), [5.])
    #     # self.assertListEqual(aero.CNB(alpha=1, mach=1).tolist(), [10.])


    # def test_aero_float_case_1(self):
    #     """test aerotable attr value usage"""
    #     aero = aerotable.AeroTable(Sref=1.23, Lref=4.56)
    #     self.assertEqual(aero.Sref, 1.23)
    #     self.assertEqual(aero.Lref, 4.56)


    # def test_aero_get_set_case_1(self):
    #     """test aerotable implicit setter / getter"""
    #     table = datatable.DataTable(self.data, self.axes)
    #     aero = aerotable.AeroTable()
    #     aero.Sref = 1.23
    #     aero.Lref = 4.56
    #     aero.CA = table
    #     self.assertEqual(aero.Sref, 1.23)
    #     self.assertEqual(aero.Lref, 4.56)
    #     self.assertListEqual(aero.CA(alpha=1, mach=1).tolist(), [5.])


    # def test_aero_increments_case_1(self):
    #     aero = aerotable.AeroTable()
    #     self.assertEqual(aero.increments, {})


    # def test_aero_increments_case_2(self):
    #     aero = aerotable.AeroTable()
    #     aero.increments = {"alpha": np.array([1., 2, 3])}
    #     self.assertEqual(aero.increments, {"alpha": [1., 2., 3.]})


if __name__ == '__main__':
    unittest.main()
