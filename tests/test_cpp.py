import unittest
import numpy as np
from japl.DataTable.DataTable import DataTable
import model
import atmosphere
import aerotable
import datatable



class TestCpp(unittest.TestCase):


    def setUp(self) -> None:
        self.TOLERANCE_PLACES = 15
        self.data = np.array([[1, 2, 3],
                              [4, 5, 6],
                              [7, 8, 9]],
                             dtype=float)
        self.axes = {"alpha": np.array([0., 1, 2]),
                     "mach": np.array([0., 1, 2])}


    def test_datatable_dict(self):
        """test call datable with dict"""
        table = datatable.DataTable(self.data, self.axes)
        ret = table(dict(alpha=1., mach=1.))
        self.assertListEqual(ret.tolist(), [5.])


    def test_datatable_kwargs(self):
        """test call datable with kwargs"""
        table = datatable.DataTable(self.data, self.axes)
        ret = table(alpha=1., mach=1.)
        self.assertListEqual(ret.tolist(), [5.])


    def test_datatable_iter(self):
        """test call datable with tuple"""
        table = datatable.DataTable(self.data, self.axes)
        ret = table([[1., 1.]])
        self.assertListEqual(ret.tolist(), [5.])


    def test_atmosphere_case_1(self):
        atmos = atmosphere.Atmosphere()
        self.assertAlmostEqual(atmos.pressure(0), 101325.0, places=self.TOLERANCE_PLACES)
        self.assertAlmostEqual(atmos.speed_of_sound(1000), 336.43458210225776,
                               places=self.TOLERANCE_PLACES)


    def test_aero_table(self):
        """test aerotable table usage"""
        table = datatable.DataTable(self.data, self.axes)
        table2 = datatable.DataTable(self.data * 2, self.axes)
        aero = aerotable.AeroTable(CA=table, CNB=table2)
        self.assertListEqual(aero.CA(alpha=1, mach=1).tolist(), [5.])
        self.assertListEqual(aero.CNB(alpha=1, mach=1).tolist(), [10.])


    def test_aero_float_case_1(self):
        """test aerotable attr value usage"""
        aero = aerotable.AeroTable(Sref=1.23, Lref=4.56)
        self.assertEqual(aero.Sref, 1.23)
        self.assertEqual(aero.Lref, 4.56)


    def test_aero_get_set_case_1(self):
        """test aerotable implicit setter / getter"""
        table = datatable.DataTable(self.data, self.axes)
        aero = aerotable.AeroTable()
        aero.Sref = 1.23
        aero.Lref = 4.56
        aero.CA = table
        self.assertEqual(aero.Sref, 1.23)
        self.assertEqual(aero.Lref, 4.56)
        self.assertListEqual(aero.CA(alpha=1, mach=1).tolist(), [5.])


    def test_aero_increments_case_1(self):
        aero = aerotable.AeroTable()
        self.assertEqual(aero.increments, {})


    def test_aero_increments_case_2(self):
        aero = aerotable.AeroTable()
        aero.increments = {"alpha": np.array([1., 2, 3])}
        self.assertEqual(aero.increments, {"alpha": [1., 2., 3.]})


    def test_model_1(self):
        """test model implicit setter / getter"""
        table = datatable.DataTable(self.data, self.axes)
        table2 = datatable.DataTable(self.data * 2, self.axes)
        aero = aerotable.AeroTable(CA=table, CNB=table2)
        m = model.Model()
        m.aerotable = aero
        self.assertListEqual(m.aerotable.CA.interp._data.tolist(), self.data.tolist())
        self.assertListEqual(m.aerotable.CNB.interp._data.tolist(), (self.data * 2).tolist())
        self.assertEqual(m.aerotable.CA_Boost.interp._data.shape, (0,))


    def test_model_2(self):
        """test model aerotable setter"""
        table = datatable.DataTable(self.data, self.axes)
        table2 = datatable.DataTable(self.data * 2, self.axes)
        aero = aerotable.AeroTable(CA=table, CNB=table2)
        m = model.Model()
        m.set_aerotable(aero)
        self.assertListEqual(m.aerotable.CA.interp._data.tolist(), self.data.tolist())
        self.assertListEqual(m.aerotable.CNB.interp._data.tolist(), (self.data * 2).tolist())
        self.assertEqual(m.aerotable.CA_Boost.interp._data.shape, (0,))


# data = np.array([[1, 2, 3],
#                  [4, 5, 6],
#                  [7, 8, 9]],
#                 dtype=float)
# axes = {"alpha": np.array([0., 1, 2]),
#         "mach": np.array([0., 1, 2])}

# table = datatable.DataTable()
# print(table.interp._data)
# table2 = datatable.DataTable(data * 2, axes)
# aero = aerotable.AeroTable(CA=table, CNB=table2)
# # aero = aerotable.AeroTable()
# atmos = atmosphere.Atmosphere()
# m = model.Model()
# # m.atmosphere = atmos
# m.aerotable = aero

if __name__ == '__main__':
    unittest.main()
