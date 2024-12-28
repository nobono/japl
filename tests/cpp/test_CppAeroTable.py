import unittest
import numpy as np
import datatable
import aerotable



class TestCppAeroTable(unittest.TestCase):


    def setUp(self) -> None:
        self.TOLERANCE_PLACES = 15
        self.data = np.array([[1, 2, 3],
                              [4, 5, 6],
                              [7, 8, 9]],
                             dtype=float)
        self.axes = {"alpha": np.array([0., 1, 2]),
                     "mach": np.array([0., 1, 2])}


    def test_load_datatables(self):
        """test aerotable table usage"""
        table = datatable.DataTable(self.data, self.axes)
        table2 = datatable.DataTable(self.data * 2, self.axes)
        aero = aerotable.AeroTable(CA=table, CNB=table2)
        self.assertListEqual(aero.CA(alpha=1, mach=1).tolist(), [5.])
        self.assertListEqual(aero.CNB(alpha=1, mach=1).tolist(), [10.])


    def test_load_scalars(self):
        """test aerotable attr value usage"""
        aero = aerotable.AeroTable(Sref=1.23, Lref=4.56)
        self.assertEqual(aero.Sref, 1.23)
        self.assertEqual(aero.Lref, 4.56)


    def test_load_values_case2(self):
        """test aerotable implicit setter / getter"""
        table = datatable.DataTable(self.data, self.axes)
        aero = aerotable.AeroTable()
        aero.Sref = 1.23
        aero.Lref = 4.56
        aero.CA = table
        self.assertEqual(aero.Sref, 1.23)
        self.assertEqual(aero.Lref, 4.56)
        self.assertListEqual(aero.CA(alpha=1, mach=1).tolist(), [5.])


    def test_aero_increments_case1(self):
        """null init"""
        aero = aerotable.AeroTable()
        self.assertEqual(aero.increments, {})

        """explicit init"""
        aero = aerotable.AeroTable()
        aero.increments = {"alpha": np.array([1., 2, 3])}
        self.assertEqual(aero.increments, {"alpha": [1., 2., 3.]})


if __name__ == '__main__':
    unittest.main()
