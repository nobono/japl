import sys
import unittest
import numpy as np
import importlib.util
import subprocess
import tempfile
from textwrap import dedent
from pathlib import Path
from japl.global_opts import get_root_dir
from japl.CodeGen.JaplFunction import JaplFunction
from japl.AeroTable.AeroTable import AeroTable
from japl.DataTable.DataTable import DataTable
from japl.Model.Model import Model


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


    # def test_set_aertotable(self):
    #     # path = Path(__file__).absolute()
    #     # source_model_path = Path(path.parent, "source_model.py")
    #     # subprocess.call(["python", source_model_path])

    #     # Step 1: Create a temporary Python file
    #     with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
    #         # contents = (b"def hello():\n"
    #         #             b"\treturn 'Hello from the temporary file!'\n")
    #         contents = (
    #                 b"from pathlib import Path"
    #                 b"from sympy import symbols, Matrix"
    #                 b"from japl.Model.Model import Model"
    #                 b""
    #                 b""
    #                 b""
    #                 b"dt = symbols('dt')"
    #                 b"input = Matrix([])"
    #                 b"state = Matrix(symbols('x, y'))"
    #                 b"model = Model.from_expression(dt_var=dt,"
    #                 b"                              input_vars=input,"
    #                 b"                              state_vars=state)"
    #                 b"model.create_c_module('simple', Path(__file__).absolute().parent)"
    #                 )
    #         temp_file.write(contents)
    #         temp_file_path = temp_file.name

    #     # print(f"Temporary Python file created at: {temp_file_path}")

    #     # Step 2: Add the directory of the temp file to sys.path for import
    #     temp_file_dir = tempfile.gettempdir()
    #     if temp_file_dir not in sys.path:
    #         sys.path.append(temp_file_dir)

    #     # Step 3: Dynamically import the temporary module
    #     module_name = temp_file_path.split("/")[-1].split(".")[0]
    #     spec = importlib.util.spec_from_file_location(module_name, temp_file_path)
    #     if spec:
    #         temp_module = importlib.util.module_from_spec(spec)
    #         sys.modules[module_name] = temp_module
    #         if spec.loader:
    #             spec.loader.exec_module(temp_module)
    #         else:
    #             raise Exception("could not load temp file")
    #     else:
    #         raise Exception("could not load temp file")

    #     # Step 4: Use the imported function
    #     # print(temp_module.hello())  # Output: Hello from the temporary file!


    #     # pyaero = AeroTable(CA=self.table)
    #     # print(id(model.input_updates))
    #     # model.set_aerotable(pyaero)
    #     # print(id(model.cpp.aerotable))


if __name__ == '__main__':
    unittest.main()
