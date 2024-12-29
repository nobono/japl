import unittest
import numpy as np
from pathlib import Path
from japl.Util.Matlab import MatFile
from japl.Util.Matlab import MatStruct
from japl.global_opts import get_root_dir



class TestMatFile(unittest.TestCase):


    def test_case1(self):
        file = MatFile(Path(get_root_dir(), "tests/unit/Util/test_file.mat"))
        attr_names = ["_raw_data",
                      "array",
                      "col_array",
                      "num_float",
                      "num_int",
                      "string",
                      "struct",
                      "cell"]
        for name in attr_names:
            self.assertTrue(hasattr(file, name))

        array = getattr(file, "array")
        col_array = getattr(file, "col_array")
        num_float = getattr(file, "num_float")
        num_int = getattr(file, "num_int")
        string = getattr(file, "string")
        struct = getattr(file, "struct")
        cell = getattr(file, "cell")
        self.assertIsInstance(array, np.ndarray)
        self.assertListEqual(array.tolist(), [1., 2., 3.])
        self.assertIsInstance(col_array, np.ndarray)
        self.assertListEqual(col_array.tolist(), [1., 2., 3.])
        self.assertEqual(num_float, 1)
        self.assertEqual(num_int, int(2))
        self.assertEqual(string, "this_string")
        self.assertEqual(len(cell), 3)
        self.assertEqual(cell[0], np.array([1]))
        self.assertEqual(cell[1], np.array(["a"]))
        self.assertTrue((cell[2] == np.array([[1., 2., 3.]])).all())
        self.assertIsInstance(struct, MatStruct)
        self.assertEqual(getattr(struct, "a"), 1)
        self.assertEqual(getattr(struct, "b"), "b")
        self.assertIsInstance(getattr(struct, "array"), np.ndarray)
        self.assertEqual(getattr(struct, "array").tolist(), [1, 2, 3])


if __name__ == '__main__':
    unittest.main()
