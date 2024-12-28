import unittest
import numpy as np
import datatable
import model
import aerotable



class TestCppModel(unittest.TestCase):


    def setUp(self) -> None:
        self.TOLERANCE_PLACES = 15
        self.data = np.array([[1, 2, 3],
                              [4, 5, 6],
                              [7, 8, 9]],
                             dtype=float)
        self.axes = {"alpha": np.array([0., 1, 2]),
                     "mach": np.array([0., 1, 2])}


    def test_model_case1(self):
        """test model implicit setter / getter"""
        table = datatable.DataTable(self.data, self.axes)
        table2 = datatable.DataTable(self.data * 2, self.axes)
        aero = aerotable.AeroTable(CA=table, CNB=table2)
        m = model.Model()
        m.aerotable = aero
        self.assertListEqual(m.aerotable.CA.interp._data.tolist(), self.data.tolist())
        self.assertListEqual(m.aerotable.CNB.interp._data.tolist(), (self.data * 2).tolist())
        self.assertEqual(m.aerotable.CA_Boost.interp._data.shape, (0,))


    def test_model_case2(self):
        """test model aerotable setter"""
        table = datatable.DataTable(self.data, self.axes)
        table2 = datatable.DataTable(self.data * 2, self.axes)
        aero = aerotable.AeroTable(CA=table, CNB=table2)
        m = model.Model()
        m.set_aerotable(aero)
        self.assertListEqual(m.aerotable.CA.interp._data.tolist(), self.data.tolist())
        self.assertListEqual(m.aerotable.CNB.interp._data.tolist(), (self.data * 2).tolist())
        self.assertEqual(m.aerotable.CA_Boost.interp._data.shape, (0,))


if __name__ == '__main__':
    unittest.main()
