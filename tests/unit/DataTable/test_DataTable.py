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
        """missing mach arg"""
        table = DataTable(self.data, self.axes)
        with self.assertRaises(Exception):
            table(alpha=1)


    def test_args(self):
        """missing mach arg"""
        table = DataTable(self.data, self.axes)
        ret = table(1, 1)
        kw_ret = table(alpha=1, mach=1)
        dict_ret = table({"alpha": 1, "mach": 1})
        self.assertEqual(ret, kw_ret)
        self.assertEqual(ret, dict_ret)


    def test_mirror_axis_case1(self):
        table = DataTable(self.data, self.axes)
        true = np.array([[-6., -7., -8.],
                         [-3., -4., -5.],
                         [0., 1., 2.],
                         [3., 4., 5.],
                         [6., 7., 8.]])
        mtable = table.mirror_axis("alpha")
        self.assertTrue((mtable == true).all())
        self.assertTrue((true == mtable.interp.interp_obj._data).all())  # type:ignore
        self.assertListEqual(mtable.axes["alpha"].tolist(), [-2, -1, 0, 1, 2])
        self.assertListEqual(mtable.axes["mach"].tolist(), self.axes["mach"].tolist())


    def test_mirror_axis_case2(self):
        table = DataTable(self.data, self.axes)
        true = np.array([[-2., -1., 0., 1., 2.],
                         [-5., -4., 3., 4., 5.],
                         [-8., -7., 6., 7., 8.]])
        mtable = table.mirror_axis("mach")
        self.assertTrue((mtable == true).all())
        self.assertTrue((true == mtable.interp.interp_obj._data).all())  # type:ignore
        self.assertListEqual(mtable.axes["mach"].tolist(), [-2, -1, 0, 1, 2])
        self.assertListEqual(mtable.axes["alpha"].tolist(), self.axes["alpha"].tolist())


    def test_add_case2(self):
        """table add"""
        t1 = DataTable(self.data, self.axes)
        t2 = DataTable(self.data, self.axes)
        true = np.array([[0., 2., 4.],
                         [6., 8., 10.],
                         [12., 14., 16.]])
        data = t1 + t2
        self.assertTrue((true == data).all())
        self.assertTrue((true == data.interp.interp_obj._data).all())  # type:ignore
        self.assertListEqual(data.axes["alpha"].tolist(), self.axes["alpha"].tolist())
        self.assertListEqual(data.axes["mach"].tolist(), self.axes["mach"].tolist())


    def test_op_align_axes(self):
        axes2 = {"alpha": np.array([0., 1, 2]),
                 "mach": np.array([0., 1, 2]),
                 "alt": np.array([0., 1])}
        data2 = np.ones((3, 2, 3))
        t1 = DataTable(self.data, self.axes)
        t2 = DataTable(data2, axes2)
        table1, table2, new_axis = DataTable._op_align_axes(t1, t2)
        true1 = [[[0.], [1.], [2.]],
                 [[3.], [4.], [5.]],
                 [[6.], [7.], [8.]]]
        true2 = [[[1., 1., 1.,],
                  [1., 1., 1.,]],
                 [[1., 1., 1.,],
                  [1., 1., 1.,]],
                 [[1., 1., 1.,],
                  [1., 1., 1.,]]]
        self.assertListEqual(table1.tolist(), true1)
        self.assertListEqual(table2.tolist(), true2)
        self.assertTrue((new_axis["alpha"] == axes2["alpha"]).all())
        self.assertTrue((new_axis["mach"] == axes2["mach"]).all())
        self.assertTrue((new_axis["alt"] == axes2["alt"]).all())


    def test_add_case3(self):
        """scalar add"""
        t1 = DataTable(self.data, self.axes)
        true = np.array([[2., 3., 4.],
                         [5., 6., 7.],
                         [8., 9., 10.]])
        data = t1 + 2
        self.assertTrue((true == data).all())
        self.assertTrue((true == data.interp.interp_obj._data).all())  # type:ignore
        self.assertListEqual(data.axes["alpha"].tolist(), self.axes["alpha"].tolist())
        self.assertListEqual(data.axes["mach"].tolist(), self.axes["mach"].tolist())


    def test_add_case4(self):
        """table add with different axes"""
        axes2 = {"alpha": np.array([0., 1, 2]),
                 "mach": np.array([0., 1, 2]),
                 "alt": np.array([0., 1])}
        data2 = np.ones((3, 3, 2))
        t1 = DataTable(self.data, self.axes)
        t2 = DataTable(data2, axes2)
        true = self.data[:, :, np.newaxis] + data2
        ret = t1 + t2
        self.assertTrue((ret == true).all())


    def test_add_case5(self):
        """table add with different axes.
        This should be same as test_add_case4 when axes
        are the same."""
        axes2 = {"alpha": np.array([0., 1, 2]),
                 "alt": np.array([0., 1]),
                 "mach": np.array([0., 1, 2]),
                 }
        data2 = np.ones((3, 2, 3))
        t1 = DataTable(self.data, self.axes)
        t2 = DataTable(data2, axes2)

        ret = t1 + t2
        self.assertEqual(ret.shape, data2.shape)
        self.assertEqual(tuple(ret.axes.keys()), tuple(axes2.keys()))


    def test_mul_case2(self):
        """table mul"""
        t1 = DataTable(self.data, self.axes)
        t2 = DataTable(self.data, self.axes)
        true = np.array([[0., 1., 4.],
                         [9., 16., 25.],
                         [36., 49., 64.]])
        data = t1 * t2
        self.assertTrue((true == data).all())
        self.assertTrue((true == data.interp.interp_obj._data).all())  # type:ignore
        self.assertListEqual(data.axes["alpha"].tolist(), self.axes["alpha"].tolist())
        self.assertListEqual(data.axes["mach"].tolist(), self.axes["mach"].tolist())


    def test_mul_case3(self):
        """table matmul"""
        t1 = DataTable(self.data, self.axes)
        t2 = DataTable(self.data, self.axes)
        true = np.array([[15., 18., 21.],
                         [42., 54., 66.],
                         [69., 90., 111.]])
        data = t1 @ t2
        self.assertTrue((true == data).all())
        self.assertTrue((true == data.interp.interp_obj._data).all())  # type:ignore
        self.assertListEqual(data.axes["alpha"].tolist(), self.axes["alpha"].tolist())
        self.assertListEqual(data.axes["mach"].tolist(), self.axes["mach"].tolist())


    def test_mul_case4(self):
        """scalar mul"""
        t1 = DataTable(self.data, self.axes)
        true = np.array([[0., 2., 4.],
                         [6., 8., 10.],
                         [12., 14., 16.]])
        data = t1 * 2
        self.assertTrue((true == data).all())
        self.assertTrue((true == data.interp.interp_obj._data).all())  # type:ignore
        self.assertListEqual(data.axes["alpha"].tolist(), self.axes["alpha"].tolist())
        self.assertListEqual(data.axes["mach"].tolist(), self.axes["mach"].tolist())


if __name__ == '__main__':
    unittest.main()
