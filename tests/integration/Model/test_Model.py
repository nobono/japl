import unittest
import numpy as np
from japl.AeroTable.AeroTable import AeroTable
from japl.DataTable.DataTable import DataTable
from japl.Model.Model import Model
from aerotable import AeroTable as CppAeroTable


class TestModel_integration(unittest.TestCase):


    def setUp(self) -> None:
        data = np.ones((2, 2), dtype=float)
        axes = ({'a': np.array([0., 1.]),
                 'b': np.array([0., 1.])})
        self.table = DataTable(data, axes)


    def test_py_to_cpp(self):
        """tests Model.set_aerotable() from py-side AeroTable argument"""
        aero = AeroTable(CA=self.table)
        model = Model()
        self.assertListEqual(model.aerotable.cpp.CA.interp._data.tolist(), [])
        model.set_aerotable(aero)
        self.assertTrue((model.aerotable.cpp.CA.interp._data == aero.CA).all())


if __name__ == '__main__':
    unittest.main()
