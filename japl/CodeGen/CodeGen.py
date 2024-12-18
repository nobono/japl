from io import TextIOWrapper
import os
import shutil
import subprocess
from pathlib import Path
from textwrap import dedent
from typing import Callable
from typing import Any, Optional, Union
from sympy.codegen.ast import Basic
from sympy.codegen.ast import Variable
from japl.global_opts import get_root_dir
from japl.CodeGen.JaplFunction import JaplFunction
from japl.CodeGen.Ast import JaplClass
from japl.CodeGen.Ast import CType
from japl.CodeGen.Ast import CTypes
from japl.CodeGen.Printer import ccode
from japl.CodeGen.Printer import pycode
from japl.CodeGen.Util import copy_dir

Writes = list[str]
AstNode = Union[str, list, tuple, Basic, "Builder"]



class Builder:

    __slots__ = ("name", "data", "code_type", "writes")
    defaults = {"writes": lambda: [], "data": lambda: {}}

    name: str
    data: dict
    code_type: str
    writes: list

    def __init__(self, name: str, contents: AstNode = [], *args, **kwargs) -> None:
        if (code_type := kwargs.get("code_type", None)) is not None:
            self.code_type = code_type
        self._apply_defaults()
        self.name = name
        self.append(contents)


    def _apply_defaults(self):
        """Checks for defaults and attempts to set any defined
        attribute defaults."""
        if not hasattr(self, "defaults"):
            return
        for key in self.__slots__:
            if (val := self.defaults.get(key, None)) is not None:
                if isinstance(val, Callable):  # type:ignore
                    setattr(self, key, val())
                else:
                    setattr(self, key, val)


    def append(self, ast_nodes: AstNode):
        # resursively append items to self.data
        if isinstance(ast_nodes, list) or isinstance(ast_nodes, tuple):
            for node in ast_nodes:
                self.append(node)
        else:
            # get node name or iterate
            if hasattr(ast_nodes, "name"):
                name = getattr(ast_nodes, "name")
            else:
                name = len(self.data)
            self.data[name] = ast_nodes


    def build(self, code_type: str) -> Writes:
        for name, ast_node in self.data.items():
            if isinstance(ast_node, str):
                self.writes += [ast_node]
            else:
                self.writes += [ast_node._build(code_type=code_type)]
        return self.writes


    def dumps(self, file: Optional[Any] = None, path: Path|str = "./",
              filename: str = ""):
        """Optional method to write build strings to file."""
        if file is None:
            _path = Path(path, filename) if filename else Path(path, self.name)
            file = open(Path(_path), "a+")
        for line in self.writes:
            file.write(line + "\n")
        file.close()


class FileBuilder(Builder):

    """Builds a file from provided filename and content.
    code_type is deduced from the extension in the filename."""

    def __init__(self, name: str, contents: AstNode = [], *args, **kwargs) -> None:
        super().__init__(name, contents, *args, **kwargs)
        self.code_type = self.deduce_code_type(name)  # deduce code type from filename extension


    def deduce_code_type(self, filename: str):
        self.check_filename(filename)
        ext = filename.split('.')[1]
        type_map = {"cpp": "c",
                    "hpp": "c",
                    "c": "c",
                    "h": "c",
                    "py": "py",
                    "pyi": "py",
                    "oct": "m",
                    "octave": "m",
                    "matlab": "m",
                    }
        if ext not in type_map:
            raise Exception(f"file extension {ext} not supported.")
        return type_map[ext]


    @staticmethod
    def check_filename(filename: str):
        if '.' not in filename:
            raise Exception("filename misssing extension.")


    def build(self, *args, **kwargs) -> Writes:
        code_type = self.code_type
        for name, ast_node in self.data.items():
            if isinstance(ast_node, str):
                self.writes += [ast_node]
            elif isinstance(ast_node, JaplFunction):
                ast_node._build(code_type=code_type)
                self.writes += [""]
                self.writes += [str(ccode(ast_node.get_def()))]
            else:
                raise Exception(f"unhandled case for type {ast_node.__class__}.")
        return self.get_header_writes() + self.writes + self.get_footer_writes()


    def get_header_writes(self) -> Writes:
        """override this method to write to top of file"""
        return []


    def get_footer_writes(self) -> Writes:
        """override this method to write to end of file"""
        return []


class Pybind:

    def __init__(self, module_name: str) -> None:
        self.module_name = module_name
        self.class_data: dict[str, list[JaplFunction]] = {}
        self.function_data: dict[str, list[JaplFunction]] = {}


    def append_method(self, method: JaplFunction):
        if method.class_name in self.class_data:
            self.class_data[method.class_name] += [method]
        else:
            self.class_data[method.class_name] = [method]


    def append_function(self, function: JaplFunction):
        if function.name in self.function_data:
            raise Exception(f"function {function.name} is already defined.")
        self.function_data[function.name] = [function]


    def build(self) -> Writes:
        pybind_writes = Pybind.get_pybind_binding_writes(module_name=self.module_name)
        class_writes = []
        for class_name, methods in self.class_data.items():
            if class_name == "Model":  # NOTE: do this better
                class_properties = ["aerotable", "atmosphere"]
            else:
                class_properties = []

            method_writes = []
            for method in methods:
                param_vars = method.get_proto().parameters
                if class_name == "Model":  # NOTE: do this better
                    param_vars = ModuleBuilder.std_args
                method_writes += Pybind.get_pybind_method_call_wrapper(func_name=method.name,
                                                                       parameters=param_vars,
                                                                       class_name=method.class_name,
                                                                       description=method.description)
            if len(method_writes):
                class_writes += Pybind.get_pybind_class_init_writes(class_name)
                class_writes += method_writes

            class_writes += Pybind.get_pybind_property_sets_gets(class_name=class_name,
                                                                 class_properties=class_properties)
            class_writes += ["\t;"]

        function_writes = []
        for func_name, functions in self.function_data.items():
            class_properties = []
            for function in functions:
                param_vars = function.get_proto().parameters
                return_type = CTypes.from_expr(function.expr)
                function_writes += Pybind.get_pybind_function_call_wrapper(func_name=function.name,
                                                                           return_type=return_type,
                                                                           parameters=param_vars,
                                                                           description=function.description)

        writes = pybind_writes + class_writes + function_writes
        writes += ["}"]
        return writes


    @staticmethod
    def get_pybind_binding_writes(module_name: str) -> Writes:
        writes = []
        writes = ["", ""]
        writes += [f"PYBIND11_MODULE({module_name}, m) " + "{"]  # }
        return writes


    @staticmethod
    def get_pybind_class_init_writes(class_name: str) -> Writes:
        writes = []
        class_bind_str = f"\tpybind11::class_<{class_name}>(m, \"{class_name}\")"
        class_constructor_str = "\t\t.def(pybind11::init<>())"
        writes += [class_bind_str]
        writes += [class_constructor_str]
        return writes


    @staticmethod
    def get_pybind_method_call_wrapper(func_name: str, parameters: tuple, class_name: str, description: str) -> Writes:
        class_def, func_name = ModuleBuilder.parse_class_func(func_name)
        # lambda wrapper to convert return of vector<> to py::array_t<>
        # _std_params_str = ", ".join([f"{typ} {var}" for typ, var in zip(ModuleBuilder.std_args_types,
        #                                                                 ModuleBuilder.std_args)])
        # _std_params_names_str = ", ".join([i for i in ModuleBuilder.std_args])
        params_signature = ", ".join([f"{ccode(var.type)} {ccode(var)}" for var in parameters])
        params_names = ", ".join([str(ccode(var)) for var in parameters])
        method_bind_str = (f"\t\t.def(\"{func_name}\",\n"
                           f"\t\t\t[]({class_name}& self, {params_signature}) -> py::array_t<double> "
                           "{\n"
                           f"\t\t\t\tvector<double> ret = self.{func_name}({params_names});\n"
                           "\t\t\t\tpy::array_t<double> np_ret(ret.size());\n"
                           "\t\t\t\tstd::copy(ret.begin(), ret.end(), np_ret.mutable_data());\n"
                           "\t\t\t\treturn np_ret;\n"
                           "\t\t\t}"
                           f", \"{description}\")")
        return [method_bind_str]


    @staticmethod
    def get_pybind_function_call_wrapper(func_name: str, return_type: CType,
                                         parameters: tuple, description: str) -> Writes:
        # lambda wrapper to convert return of vector<> to py::array_t<>
        if return_type.is_array:
            params_signature = ", ".join([f"{ccode(var.type)} {ccode(var)}" for var in parameters])
            params_names = ", ".join([str(ccode(var)) for var in parameters])
            method_bind_str = (f"\t\tm.def(\"{func_name}\",\n"
                               f"\t\t\t[]({params_signature}) -> py::array_t<double> "
                               "{\n"
                               f"\t\t\t\tvector<double> ret = {func_name}({params_names});\n"
                               "\t\t\t\tpy::array_t<double> np_ret(ret.size());\n"
                               "\t\t\t\tstd::copy(ret.begin(), ret.end(), np_ret.mutable_data());\n"
                               "\t\t\t\treturn np_ret;\n"
                               "\t\t\t}"
                               f", \"{description}\")")
        else:
            method_bind_str = f"\tm.def(\"{func_name}\", &{func_name}, \"{description}\");"
        return [method_bind_str]


    @staticmethod
    def get_pybind_property_sets_gets(class_name: str, class_properties: list) -> Writes:
        # write setters / getters for class properties
        ret = []
        for property in class_properties:
            gets_sets = (f"\t\t.def_property(\"{property}\",\n"
                         f"\t\t\t[](const {class_name}& self) -> const decltype({class_name}::{property})& "
                         "{"
                         f"return self.{property}" + ";},\n"
                         f"\t\t\t[]({class_name}& self, const decltype({class_name}::{property})& value) "
                         "{" + f"self.{property}" + " = value;})")
            ret += [gets_sets]
        return ret


class CFileBuilder(FileBuilder):

    __slots__ = FileBuilder.__slots__ + ("class_name", "class_properties")
    defaults = {**FileBuilder.defaults, "class_name": "Model",
                "class_properties": lambda: [], "code_type": "c"}

    class_name: str
    class_properties: list[str]


    def get_header_writes(self) -> Writes:
        """override this method to write to top of file"""
        header = ["#include <iostream>",
                  "#include <model.hpp>",
                  "#include <vector>",
                  "#include <pybind11/pybind11.h>",
                  "#include <pybind11/numpy.h>",
                  "#include <pybind11/stl.h>  // Enables automatic conversion",
                  "",
                  "namespace py = pybind11;",
                  "using std::vector;",
                  "",
                  ""]
        return header


    def get_footer_writes(self) -> Writes:
        """override this method to write to end of file"""
        if '.' in self.name:
            module_name = self.name.split('.')[0]
        else:
            module_name = self.name
        return self.get_pybind_writes(module_name=module_name,
                                      class_name=self.class_name,
                                      class_properties=self.class_properties)


    def get_pybind_writes(self, module_name: str, class_name: str, class_properties: list[str]) -> Writes:
        pybind = Pybind(module_name)
        for ast_node in self.data.values():
            if isinstance(ast_node, JaplFunction):
                if ast_node.class_name:
                    pybind.append_method(ast_node)
                else:
                    pybind.append_function(ast_node)
        writes = pybind.build()
        return writes


class ModuleBuilder(Builder):

    data: dict[str, FileBuilder]
    std_args = (Variable("t", type=CType("double").as_ref()),
                Variable("_X_arg", type=CType("double").as_vector().as_ref()),
                Variable("_U_arg", type=CType("double").as_vector().as_ref()),
                Variable("_S_arg", type=CType("double").as_vector().as_ref()),
                Variable("dt", type=CType("double").as_ref()))
    JAPL_EXT_MODULE_INIT_HEADER__ = "# __JAPL_EXTENSION_MODULE__\n"
    CXX_STD = 17

    class_properties = ["aerotable", "atmosphere"]

    def __init__(self, name: str, contents: FileBuilder|list|tuple = [], *args, **kwargs) -> None:
        super().__init__(name, contents, *args, **kwargs)


    @staticmethod
    def check_filename(filename: str):
        if '.' not in filename:
            raise Exception("filename misssing extension.")


    def build(self, *args, **kwargs) -> Writes:
        for name, file_node in self.data.items():
            # add class_properties to source FileBuilders here
            setattr(file_node, "class_properties", self.class_properties)  # NOTE: make this better
            # get code_type for FileBulder
            self.writes += file_node.build()
        return self.writes


    @staticmethod
    def create_module_directory(name: str, path: str) -> Path:
        # create extension module directory
        module_dir_path = Path(path, name)
        if os.path.exists(module_dir_path):
            input_str = f"{module_dir_path} already exists. Overwrite? (y/n):"
            if input(input_str).strip().lower() == "y":
                shutil.rmtree(module_dir_path)
            else:
                print("exiting.")
                quit()
        os.mkdir(module_dir_path)
        return module_dir_path


    def create_init_file_builder(self):
        # create __init__.py file
        module_name = self.name
        contents = ([self.JAPL_EXT_MODULE_INIT_HEADER__,
                     "import linterp",
                     "import datatable",
                     "import aerotable",
                     "import atmosphere",
                     f"from {module_name}.{module_name} import *\n"])
        init_builder = FileBuilder("__init__.py", contents=contents)
        return init_builder


    @staticmethod
    def parse_class_func(func_name: str) -> tuple[str, str]:
        # handle func_name references class method "Class::method"
        class_ref = ""
        if "::" in func_name:
            _func_str_split = func_name.split("::")
            class_ref = "".join(_func_str_split[0])
            func_name = "".join(_func_str_split[1:])
        return class_ref, func_name


    @staticmethod
    def create_build_file_builder(module_name: str, module_dir_path: str|Path, source_file: str):
        file_name = source_file.split('.')[0]

        build_str = ("""\
        import os
        import sys
        import glob
        import shutil
        from setuptools import setup
        from setuptools.command.build_ext import build_ext
        from setuptools import Command
        from pybind11.setup_helpers import Pybind11Extension
        from pathlib import Path
        import importlib.util
        import sysconfig



        def get_root_dir():
            # Find the module spec for the japl package
            spec = importlib.util.find_spec("japl")
            if spec is None or spec.origin is None:
                raise RuntimeError("japl package is not installed or cannot be found.")
            # Get the root directory of the japl package
            root_dir = os.path.dirname(spec.origin)
            return os.path.dirname(root_dir)


        dir = os.path.dirname(__file__)
        install_dir = get_root_dir()

        if not os.path.isdir(install_dir):
            raise Exception("cannot build. required package, japl, is not installed.")

        # Default build ops
        if len(sys.argv) == 1:
            sys.argv.append("build_ext")
            sys.argv.append("--build-lib")
            sys.argv.append(dir)


        class CleanCommand(Command):
            \"\"\"Custom clean command to tidy up the project root.\"\"\"
            user_options = []

            def initialize_options(self):
                pass

            def finalize_options(self):
                pass

            def run(self):
                shutil.rmtree('./build', ignore_errors=True)
                shutil.rmtree('./dist', ignore_errors=True)
                root_path = os.path.dirname(__file__)
                file_patterns = ["*.so", "*.dll"]
                for pattern in file_patterns:
                    for file in glob.iglob(str(Path(root_path, "**", pattern)), recursive=True):
                        print("removing:", file)
                        os.remove(file)


        """f"""
        sources = [str(Path(dir, "{source_file}"))]

        # Define extension module
        ext_module = Pybind11Extension(name="{module_name}",
        """"""
                                       sources=sources,
                                       extra_compile_args=[],
                                       extra_link_args=[f"{install_dir}/libs/src/linterp/linterp.o",
                                                        f"{install_dir}/libs/src/datatable.o",
                                                        f"{install_dir}/libs/src/atmosphere_alts.o",
                                                        f"{install_dir}/libs/src/atmosphere_density.o",
                                                        f"{install_dir}/libs/src/atmosphere_grav_accel.o",
                                                        f"{install_dir}/libs/src/atmosphere_pressure.o",
                                                        f"{install_dir}/libs/src/atmosphere_temperature.o",
                                                        f"{install_dir}/libs/src/atmosphere_speed_of_sound.o",
                                                        f"{install_dir}/libs/src/atmosphere.o",
                                                        f"{install_dir}/libs/src/aerotable.o",
                                                        f"{install_dir}/libs/src/model.o"],
                                       include_dirs=[Path(install_dir, "include")],
        """f"""
                                       cxx_std={ModuleBuilder.CXX_STD})
        """"""

        cmdclass = {'build_ext': build_ext,
                    'clean': CleanCommand}

        """f"""
        # Build the extension
        setup(
            name="{file_name}",
            ext_modules=[ext_module],
            cmdclass=cmdclass,
            # script_args=["build_ext", "--build-lib", dir]
        )
        """)

        build_builder = FileBuilder("build.py", dedent(build_str))
        return build_builder


class CodeGenerator:


    @staticmethod
    def build_c_module(builder: ModuleBuilder, other_builders: list[FileBuilder] = []):
        name = builder.name
        filename = name + ".cpp"
        module_dir_path = builder.create_module_directory(name=name, path="./")
        init_file_builder = builder.create_init_file_builder()
        build_file_builder = builder.create_build_file_builder(module_name=name,
                                                               module_dir_path=module_dir_path,
                                                               source_file=filename)
        builder.build()  # source files data within ModuleBuilder
        init_file_builder.build()
        build_file_builder.build()
        builder.dumps(path=module_dir_path, filename=filename)
        init_file_builder.dumps(path=module_dir_path)
        build_file_builder.dumps(path=module_dir_path)

        for blder in other_builders:
            blder.build()
            blder.dumps(path=module_dir_path)

        # copy over japl libs
        # NOTE: may not need to do this anymore
        # CodeGenerator.copy_japl_libs_to(module_dir_path)

        # try to build
        try:
            subprocess.run(["python", Path(module_dir_path, "build.py")])
        except Exception as e:
            print("Error building model", e)


    @staticmethod
    def copy_japl_libs_to(path: Path|str):
        try:
            os.mkdir(Path(path, "libs"))
        except Exception as e:
            print("Error moving libs to model dir", e)
        copy_dir(Path(get_root_dir(), "libs"), Path(path, "libs"))


    @staticmethod
    def build_file(builder: FileBuilder, file: Optional[Any] = None):
        CodeGenerator.check_for_code_type(builder)
        code_type = builder.code_type
        builder.build(code_type)
        if file:
            builder.dumps(file=file)
        else:
            file = CodeGenerator.create_file(name=builder.name, path="./")
            builder.dumps(file=file)


    @staticmethod
    def create_file(name: str, path: str):
        file_path = Path(path, name)
        if os.path.exists(file_path) and os.path.isfile(file_path):
            input_str = f"{file_path} already exists. Overwrite? (y/n):"
            if input(input_str).strip().lower() == "y":
                os.remove(file_path)
            else:
                print("exiting.")
                quit()
        return open(file_path, "a+")


    @staticmethod
    def check_for_code_type(builder: Builder) -> None:
        if not hasattr(builder, "code_type"):
            raise Exception("Builder class has no defined output language "
                            "ensure attribute \"code_type\" is defined.")
