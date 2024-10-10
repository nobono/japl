import os
import sys
import numpy as np
from japl.BuildTools.CodeGeneratorBase import CodeGeneratorBase
from sympy import ccode
from sympy import Expr
from sympy import Matrix
from sympy import symbols
from sympy import cse
from sympy.codegen.ast import float32, real
from textwrap import dedent



class CCodeGenerator(CodeGeneratorBase):

    comment_prefix: str = "//"
    pre_bracket: str = "["
    post_bracket: str = "]"
    bracket_separator: str = "]["  # ]
    endl: str = ";\n"

    header: list[str] = ["#include <pybind11/pybind11.h>",
                         "#include <pybind11/numpy.h>",
                         "#include <pybind11/stl.h>  // Enables automatic conversion",
                         "",
                         "namespace py = pybind11;",
                         ""]

    def __init__(self, strict: bool = False):
        self.strict = strict
        self.function_register = {}


    def get_code(self, expression):
        return ccode(expression, type_aliases={real: float32}, strict=self.strict)


    def write_subexpressions(self, subexpressions):
        ret = ""
        for (lvalue, rvalue) in subexpressions:
            assign_str = (self.declare_parameter(lvalue, const=True)
                          + " = " + self.get_code(rvalue) + ";")  # type:ignore
            ret += self.indent_lines(assign_str)
        return ret


    def write_matrix(self,
                     matrix,
                     variable,
                     is_symmetric=False):
        write_string = ""
        variable_name = self.get_expr_name(variable)

        ############################
        # L-value / R-value assign
        ############################
        # if Matrix of single expression
        if matrix.shape[0] * matrix.shape[1] == 1:
            write_string = write_string + variable_name + " = " + self.get_code(matrix[0]) + self.endl  # type:ignore
        # if row or column Matrix
        elif matrix.shape[0] == 1 or matrix.shape[1] == 1:
            for i in range(0, len(matrix)):
                write_string = write_string + variable_name +\
                               self.pre_bracket +\
                               str(i) +\
                               self.post_bracket + " = " +\
                               self.get_code(matrix[i]) + self.endl  # type:ignore
        # if any other shape
        else:
            for j in range(0, matrix.shape[1]):
                for i in range(0, matrix.shape[0]):
                    if j >= i or not is_symmetric:
                        write_string = write_string + variable_name +\
                                       self.pre_bracket +\
                                       str(i) + self.bracket_separator + str(j) +\
                                       self.post_bracket + " = " +\
                                       self.get_code(matrix[i, j]) + self.endl  # type:ignore

        return self.indent_lines(write_string)


    def write_function_definition(self, name: str, params: list[Expr], **kwargs):
        func_return_type = "py::array_t<double>"
        params_list, params_unpack = self.get_function_parameters(params)
        func_proto = f"{func_return_type} {name}({params_list})" + " {\n\n"  # }
        return func_proto + self.indent_lines(params_unpack)


    def write_function_returns(self, expr: Expr|Matrix, return_names: list[str]):
        if len(return_names) > 1:
            raise Exception("CCodeGenerator currently only supports returns of a"
                            "single object.")
        return_name = return_names[0]
        if self.is_array_type(expr):
            # convert vector to array_t
            return_str = "py::array_t<double>(" + return_name + ".size(), " + return_name + ".data())"
            return self.indent_lines(f"return {return_str}" + self.endl) + "}"
        else:
            return self.indent_lines(f"return {return_name}" + self.endl) + "}"


    def declare_parameter(self, param: Expr, name: str = "", const: bool = False) -> str:
        if name:
            param_name = name
        else:
            param_name = self.get_expr_name(param)

        # handle argument being an array
        if self.is_array_type(param):
            param_str = f"std::vector<double> {param_name}"
            if const:
                param_str += "const " + param_str
        else:
            param_str = f"double {param_name}"
        return param_str


    def instantiate_return_array(self, param: Expr|Matrix, name: str) -> str:
        ret = ""

        # handle argument being an array
        if self.is_array_type(param):
            shape = param.shape  # type:ignore
            shape_str = ", ".join([str(i) for i in shape])
            if len(shape) > 1:
                if shape[1] == 1:
                    constructor_str = "(" + str(shape[0]) + ")"
                else:
                    constructor_str = "({" + shape_str + "})"
            else:
                constructor_str = "(" + str(shape[0]) + ")"
            param_str = f"std::vector<double> {name}" + constructor_str
        else:
            param_str = f"double {name}"
        ret += self.indent_lines(param_str + ";")

        # write code to get pointer from pybind11 buffer info
        # request_str = f"py::buffer_info buf_info = {return_name}.request();"
        # cast_str = f"double* {param_name} = static_cast<double*>(buf_info.ptr);"
        # self.write_lines(request_str, prefix="\t")
        # self.write_lines(cast_str, prefix="\t")
        # self.write_lines("")

        # declare py::array_t and convert return var (vector) to array_t
        # if self.is_array_type(param):
        #     constructor_str = f"({param_name}.size(), {param_name}.data())"
        #     return_array = f"py::array_t<double> {name}{constructor_str}"
        #     # self.write_lines(return_array, prefix="\t")
        #     ret += self.indent_lines(return_array)
        # else:
        #     pass

        return ret


    def register_function(self, name: str, writes: list[str], description: str = ""):
        self.function_register.update({name: {"writes": writes, "description": description}})


    def add_function(self,
                     expr: Expr|Matrix,
                     params: list,
                     function_name: str,
                     return_name: str,
                     is_symmetric: bool = False):
        expr_replacements, expr_simple = cse(expr, symbols("X_temp0:1000"), optimizations='basic')
        func_def = self.write_function_definition(name=function_name,
                                                  params=params,
                                                  returns=[return_name])
        return_array = self.instantiate_return_array(param=expr, name=return_name)
        sub_expr = self.write_subexpressions(expr_replacements)
        func_body = self.write_matrix(matrix=Matrix(expr_simple),
                                      variable=return_name,
                                      is_symmetric=is_symmetric)
        func_ret = self.write_function_returns(expr=expr, return_names=[return_name])

        writes = [
                  func_def,
                  return_array,
                  sub_expr,
                  func_body,
                  func_ret,
                  "",
                  ]

        self.register_function(name=function_name, writes=writes)


    def create_module(self, module_name: str, path: str):
        # create extension module directory
        module_dir_name = module_name + "_model"
        module_dir_path = os.path.join(path, module_dir_name)
        os.mkdir(module_dir_path)

        self.path = path
        self.file_name = module_name + ".cpp"
        file_path = os.path.join(module_dir_path, self.file_name)
        self.file = open(file_path, "w")

        header = self.write_header()
        self.write_lines(header)

        pybind_writes = ["", "", f"PYBIND11_MODULE({module_name}, m) " + "{"]  # }

        # get functions from register
        for func_name, info in self.function_register.items():
            writes = info["writes"]
            description = info["description"]
            pybind_writes += [f"\tm.def(\"{func_name}\", &{func_name}, \"{description}\");"]
            for line in writes:
                self.write_lines(line)

        pybind_writes += ["}"]

        for line in pybind_writes:
            self.write_lines(line)

        self.create_build_file(path=module_dir_path,
                               module_name=module_name,
                               source=self.file_name)

        self.close()


    def create_build_file(self, module_name: str, path: str, source: str):
        # from pybind11.setup_helpers import Pybind11Extension
        # numpy_includes = ", ".join(np.get_include())

        build_str = (f"""\
        import os
        from setuptools import setup
        from setuptools.command.build_ext import build_ext
        from pybind11.setup_helpers import Pybind11Extension



        dir = os.path.dirname(__file__)
        sources = [os.path.join(dir, "{source}")]

        # Define extension module
        ext_module = Pybind11Extension(name="{module_name}",
                                       sources=sources,
                                       extra_compile_args=['-std=c++11'],
                                       extra_link_args=['-std=c++11'])
        """"""
        # Build the extension
        setup(
            ext_modules=[ext_module],
            cmdclass={'build_ext': build_ext}
        )
        """)

        build_file_name = "build.py"
        build_file_path = os.path.join(path, build_file_name)
        with open(build_file_path, "a+") as f:
            f.write(dedent(build_str))


    def close(self):
        self.file.close()
