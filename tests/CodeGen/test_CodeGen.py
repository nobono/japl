import unittest
from sympy import symbols
from japl.CodeGen.Util import ccode
from japl.CodeGen.CodeGen import Builder
from japl.CodeGen.CodeGen import FileBuilder
from japl.CodeGen.CodeGen import CFileBuilder
from japl.CodeGen.CodeGen import ModuleBuilder
from japl.CodeGen.CodeGen import CodeGenerator
from japl.CodeGen.JaplFunction import JaplFunction
from io import StringIO



a, b = symbols("a, b")


class func(JaplFunction):
    expr = a + b


class TestTemplate(unittest.TestCase):


    def setUp(self) -> None:
        pass


    def test_Builder_case1(self):
        builder = Builder("test")
        self.assertEqual(builder.name, "test")
        self.assertEqual(builder.data, {})
        self.assertEqual(builder.writes, [])

        builder = Builder("test")
        builder.append(["line1"])
        builder.append(["line2"])
        self.assertEqual(builder.data, {0: "line1", 1: "line2"})

        builder = Builder("test", ["line3", "line4"])
        self.assertEqual(builder.data, {0: "line3", 1: "line4"})

        builder = Builder("test", ["line3", "line4"])
        writes = builder.build(code_type='c')
        self.assertEqual(writes, ["line3", "line4"])


    def test_FileBuilder_case1(self):
        f = func(a, b)
        builder = FileBuilder("test.cpp", [f, "line1"])
        self.assertEqual(builder.name, "test.cpp")
        self.assertEqual(builder.data, {f.name: f, 1: "line1"})
        builder.build()

    def test_CFileBuilder_case1(self):
        f = func(a, b)
        builder = CFileBuilder("test.cpp", [f])
        self.assertEqual(builder.data, {f.name: f})
        builder.build()


    def test_ModuleBuilder_case1(self):
        builder = ModuleBuilder("mod")
        self.assertEqual(builder.name, "mod")
        self.assertEqual(builder.writes, [])

        file_builder = FileBuilder("test.cpp", [func(a, b)])
        builder = ModuleBuilder("mod", [file_builder])
        self.assertEqual(builder.name, "mod")
        self.assertEqual(builder.writes, [])
        self.assertEqual(builder.data, {file_builder.name: file_builder})
        writes = builder.build()


    # def test_CodeGenerator_c_module_case1(self):
    #     file_builder = CFileBuilder("test.cpp", [func(a, b)])
    #     builder = ModuleBuilder("mod", [file_builder])
    #     CodeGenerator.build_c_module(builder)


    # def test_CodeGenerator_case1(self):
    #     a, b = symbols("a, b")
    #     class func(JaplFunction):  # type:ignore # noqa
    #         expr = a + b
    #     f = func(a, b)

    #     builder = CFileBuilder("test.cpp")
    #     builder.add_function(f)
    #     file = StringIO()
    #     CodeGenerator.build_file(builder, file=file)


if __name__ == '__main__':
    unittest.main()
