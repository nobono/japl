import unittest
import numpy as np
from japl.DataTable.DataTable import DataTable



class TestDataTable(unittest.TestCase):


    def setUp(self) -> None:
        pass


    def test_case1(self):
        axes = {"alpha": np.array([0., 1, 2]),
                "mach": np.array([0., 1, 2])}
        data = np.array([[0, 1, 2],
                         [3, 4, 5],
                         [6, 7, 8]], dtype=float)
        table = DataTable(data, axes)
        # self.assertEqual(table(alpha=1, mach=2), 5.0)
        print(table(phi=1, mach=2))


if __name__ == '__main__':
    unittest.main()
