import linterp
import unittest
import numpy as np
import datatable
import aerotable
from japl.AeroTable.AeroTable import AeroTable
from japl.DataTable.DataTable import DataTable



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


    def test_aero_increments_from_pyside(self):
        """tests increments in cpp-class set from py-class"""
        pytable = DataTable(self.data, self.axes)
        pyaero = AeroTable(CA=pytable, increments=self.axes)
        # pyaero.increments.alpha = self.axes["alpha"]
        self.assertTrue((pyaero.increments.to_dict()["alpha"] == self.axes["alpha"]).all())
        self.assertTrue((pyaero.increments.to_dict()["mach"] == self.axes["mach"]).all())
        self.assertListEqual(pyaero.cpp.increments["alpha"], self.axes["alpha"].tolist())
        self.assertListEqual(pyaero.cpp.increments["mach"], self.axes["mach"].tolist())


    def test_stages(self):
        stage1 = aerotable.AeroTable(Sref=1.0)
        stage2 = aerotable.AeroTable(Sref=2.0)
        aero = aerotable.AeroTable()
        aero.add_stage(stage1)
        aero.add_stage(stage2)
        self.assertEqual(len(aero.stages), 2)
        self.assertIsInstance(aero.stages[0], aerotable.AeroTable)
        self.assertIsInstance(aero.stages[1], aerotable.AeroTable)
        self.assertEqual(aero.stages[0].is_stage, True)
        self.assertEqual(aero.stages[1].is_stage, True)
        self.assertEqual(aero.stage_id, 0)
        self.assertEqual(aero.is_stage, False)
        self.assertEqual(stage1.is_stage, True)
        self.assertEqual(stage2.is_stage, True)


if __name__ == '__main__':
    unittest.main()
