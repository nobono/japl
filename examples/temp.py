from sympy import symbols
from sympy import Matrix
from japl.CodeGen import pycode
from japl.CodeGen import ccode
from japl.CodeGen import Builder
from japl.CodeGen import FileBuilder
from japl.CodeGen import CFileBuilder
from japl.CodeGen import ModuleBuilder
from japl.CodeGen import CodeGenerator
from japl.CodeGen import JaplFunction


a, b, c = symbols("a, b, c")


class func(JaplFunction):
    class_name = "Model"
    expr = a + b


class func2(JaplFunction):
    class_name = "Model"
    expr = 2 * a + (b / 4.) + c


f = func(a, b)
f2 = func2(a, b, c)

# cd = ast.Assignment(c, func(a, b))
# print(pycode(cd))
# print(ccode(cd))

file_builder = CFileBuilder("mod.cpp", [f, f2])
builder = ModuleBuilder("mod", [file_builder])
CodeGenerator.build_c_module(builder)

# print(b.__slots__)
# print(a.__slots__)

# source = CFileBuilder("test.cpp", [f, f2])
# ext_module = ModuleBuilder("exttest", [source])
# CodeGenerator.build_c_module(ext_module)

# ext_module.append()
# CodeGenerator.build_file(source)


# # create __init__.py file
# with open(os.path.join(module_dir_path, "__init__.py"), "a+") as f:
#     f.write(self.JAPL_EXT_MODULE_INIT_HEADER__)
#     f.write("import linterp\n")
#     f.write("import datatable\n")
#     f.write("import aerotable\n")
#     f.write("import atmosphere\n")
#     f.write(f"from {module_name}.{module_name} import *\n")
