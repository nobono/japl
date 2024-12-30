import unittest
import numpy as np
from japl.AeroTable.AeroTable import AeroTable
from japl.DataTable.DataTable import DataTable
from japl.Model.Model import Model


class TestModel_integration(unittest.TestCase):

    def test_py_to_cpp(self):
        """tests Model.set_aerotable() from py-side AeroTable argument"""
        data = np.ones((2, 2))
        axes = ({'a': np.array([0., 1.]),
                 'b': np.array([0., 1.])})
        table = DataTable(data, axes)
        aero = AeroTable(CA=table)
        model = Model()
        self.assertListEqual(model.aerotable.CA.interp._data.tolist(), [])
        model.set_aerotable(aero)
        self.assertTrue((model.aerotable.CA.interp._data == aero.CA).all())


if __name__ == '__main__':
    unittest.main()
