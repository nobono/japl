import unittest
from sympy import symbols
from japl.CodeGen.Util import ccode
from japl.CodeGen.CodeGen import Builder
from japl.CodeGen.CodeGen import FileBuilder
from japl.CodeGen.CodeGen import CFileBuilder
from japl.CodeGen.CodeGen import CodeGenerator
from japl.CodeGen.JaplFunction import JaplFunction
from io import StringIO



# TODO: these tests dont assert anything



class TestTemplate(unittest.TestCase):


    def setUp(self) -> None:
        pass


    def test_filebuilder_case1(self):
        code_type = "c"
        a, b = symbols("a, b")
        class func(JaplFunction):  # type:ignore # noqa
            expr = a + b
        f = func(a, b)
        builder = FileBuilder("test.cpp")
        builder.add_function(f)
        builder.build(code_type)

        file = StringIO()
        builder.dumps(file)

        file.seek(0)
        code_str = file.read()
        file.close()


    def test_CodeGenerator_case1(self):
        a, b = symbols("a, b")
        class func(JaplFunction):  # type:ignore # noqa
            expr = a + b
        f = func(a, b)

        builder = CFileBuilder("test.cpp")
        builder.add_function(f)
        file = StringIO()
        CodeGenerator.build_file(builder, file=file)


if __name__ == '__main__':
    unittest.main()
