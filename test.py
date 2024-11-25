import unittest
import numpy as np
from japl.DataTable.DataTable import DataTable
import model
import aerotable
import datatable



data = np.array([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]],
                dtype=float)
axes = {"alpha": np.array([0., 1, 2]),
        "mach": np.array([0., 1, 2])}



class TestCpp(unittest.TestCase):


    def setUp(self) -> None:
        pass


    def test_datatable(self):
        table = datatable.DataTable(data, axes)
        ret = table(alpha=1, mach=1)
        self.assertListEqual(ret, [5.])

    def test_aero_1(self):
        table = datatable.DataTable(data, axes)
        table2 = datatable.DataTable(data * 2, axes)
        aero = aerotable.AeroTable(CA=table, CNB=table2)
        self.assertListEqual(aero.CA(alpha=1, mach=1), [5.])
        self.assertListEqual(aero.CNB(alpha=1, mach=1), [10.])


if __name__ == '__main__':
    unittest.main()
