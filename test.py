import unittest
import numpy as np
from japl.DataTable.DataTable import DataTable
# import model
import aerotable
import datatable



# data = np.array([[1, 2, 3],
#                  [4, 5, 6],
#                  [7, 8, 9]],
#                 dtype=float)
# axes = {"alpha": np.array([0., 1, 2]),
#         "mach": np.array([0., 1, 2])}



class TestCpp(unittest.TestCase):


    def setUp(self) -> None:
        self.data = np.array([[1, 2, 3],
                              [4, 5, 6],
                              [7, 8, 9]],
                             dtype=float)
        self.axes = {"alpha": np.array([0., 1, 2]),
                     "mach": np.array([0., 1, 2])}

    # def test_cc(self):
    #     table = datatable.DataTable(self.data, self.axes)
    #     # print(table.interp((1, 1)))
    #     table.cc_test()
    #     table.cc_test()
    #     # print(table.interp2((1, 0)))

    def test_datatable_dict(self):
        table = datatable.DataTable(self.data, self.axes)
        ret = table(dict(alpha=1., mach=1.))
        self.assertListEqual(ret, [5.])

    def test_datatable_kwargs(self):
        table = datatable.DataTable(self.data, self.axes)
        ret = table(alpha=1., mach=1.)
        self.assertListEqual(ret, [5.])

    def test_datatable_iter(self):
        table = datatable.DataTable(self.data, self.axes)
        ret = table([[1., 1.]])
        self.assertListEqual(ret, [5.])

    def test_aero_1(self):
        table = datatable.DataTable(self.data, self.axes)
        table2 = datatable.DataTable(self.data * 2, self.axes)
        aero = aerotable.AeroTable(CA=table, CNB=table2)
        self.assertListEqual(aero.CA(alpha=1, mach=1), [5.])
        self.assertListEqual(aero.CNB(alpha=1, mach=1), [10.])

    # def test_model_1(self):
    #     table = datatable.DataTable(data, axes)
    #     table2 = datatable.DataTable(data * 2, axes)
    #     aero = aerotable.AeroTable(CA=table, CNB=table2)
    #     m = model.Model()
    #     m.set_aerotable(aero)
    #     print(m.aerotable.CNB._data)
    #     print(aero.CNB._data)


if __name__ == '__main__':
    unittest.main()
