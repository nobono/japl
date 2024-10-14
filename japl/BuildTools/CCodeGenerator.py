import os
import sys
import shutil
from sympy.matrices.expressions.matexpr import MatrixSymbol
from tqdm import tqdm
import numpy as np
from japl.BuildTools.CodeGeneratorBase import CodeGeneratorBase
from sympy import ccode
from sympy import Expr
from sympy import Matrix
from sympy import Symbol
from sympy import symbols
from sympy.codegen.ast import float64, real
from textwrap import dedent
from japl.BuildTools.BuildTools import parallel_subs
from japl.BuildTools.BuildTools import dict_subs_func
from japl.BuildTools.BuildTools import parallel_cse
from collections import defaultdict



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
        return ccode(expression, type_aliases={real: float64}, strict=self.strict)


    def write_subexpressions(self, subexpressions: tuple|list) -> str:
        ret = ""
        for (lvalue, rvalue) in subexpressions:
            # assign_str = (self.declare_variable(lvalue, prefix="const")
            #               + " = " + self.get_code(rvalue) + self.endl)  # type:ignore
            assign_str = self._declare_variable(lvalue, prefix="const", assign=self.get_code(rvalue))  # type:ignore
            ret += assign_str
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
            # for j in range(0, matrix.shape[1]):
            #     for i in range(0, matrix.shape[0]):
            #         if j >= i or not is_symmetric:
            #             write_string = write_string + variable_name +\
            #                            self.pre_bracket +\
            #                            str(i) + self.bracket_separator + str(j) +\
            #                            self.post_bracket + " = " +\
            #                            self.get_code(matrix[i, j]) + self.endl  # type:ignore
            ###################################################
            for i in range(0, matrix.shape[0]):
                for j in range(0, matrix.shape[1]):
                    # flatten the array
                    if is_symmetric and (i < j):
                        index = j + i * matrix.shape[0]
                    else:
                        index = i + j * matrix.shape[1]

                    write_string = (write_string + variable_name
                                    + self.pre_bracket
                                    + str(index)
                                    + self.post_bracket + " = "
                                    + self.get_code(matrix[i, j]) + self.endl)  # type:ignore

        return write_string


    @staticmethod
    def _get_return_type(expr) -> str:
        # determine return shape
        return_shape = Matrix([expr]).shape

        if np.prod(return_shape) == 1:
            primitive_type = CCodeGenerator._get_primitive_type(expr)  # type:ignore
            type_str = primitive_type
        else:
            primitive_type = "double"
            type_str = f"py::array_t<{primitive_type}>"
        return type_str


    def write_function_definition(self, name: str, expr: Expr|Matrix, params: list[Expr|Symbol|Matrix]):
        return_type_str = self._get_return_type(expr)
        params_list, params_unpack = self.get_function_parameters(params)
        func_proto = f"{return_type_str} {name}({params_list})" + " {\n"  # }
        return func_proto + self.indent_lines(params_unpack)


    def write_function_returns(self, expr: Expr|Matrix, return_names: list[str]):
        if len(return_names) > 1:
            raise Exception("CCodeGenerator currently only supports returns of a"
                            "single object.")
        return_name = return_names[0]
        if self.is_array_type(expr):
            # convert vector to array_t
            return_str = "py::array_t<double>(" + return_name + ".size(), " + return_name + ".data())"
            return f"return {return_str}" + self.endl
        else:
            return f"return {return_name}" + self.endl


    @staticmethod
    def _get_primitive_type(param: Expr) -> str:
        if param.assumptions0.get("integer", False) or getattr(param, "is_integer"):
            primitive_type = "int"
        elif param.assumptions0.get("boolean", False):
            primitive_type = "bool"
        else:
            primitive_type = "double"
        return primitive_type


    @staticmethod
    def _get_type(param: Expr|Matrix, prefix: str = "") -> str:
        # handle argument being an array or not
        if CodeGeneratorBase.is_array_type(param):
            primitive_type = "double"  # NOTE: currently default for arrays is double
            type_str = f"{prefix} std::vector<{primitive_type}>"
        else:
            primitive_type = CCodeGenerator._get_primitive_type(param)  # type:ignore
            type_str = f"{prefix} {primitive_type}"
        return type_str.strip(" ")


    @staticmethod
    def _assign_variable(lvalue: str, rvalue: str, op: str = "=") -> str:
        return f"{lvalue} {op} {rvalue}" + CCodeGenerator.endl


    @staticmethod
    def _declare_variable(param: Expr|Matrix, force_name: str = "", assign: str = "", prefix: str = "") -> str:
        if not force_name:
            CodeGeneratorBase._raise_exception_non_variable(param)

        type_str = CCodeGenerator._get_type(param)

        if CodeGeneratorBase.is_array_type(param):
            if not force_name:
                raise Exception("declaration of Matrix requires \"force_name\" parameter")
            size = np.prod(param.shape)  # type:ignore
            constructor_str = f"({size})"
            if assign:
                param_str = CCodeGenerator._assign_variable(f"{prefix} {type_str} {force_name}{constructor_str}",
                                                            assign, op="=")
            else:
                param_str = f"{prefix} {type_str} {force_name}{constructor_str}" + CCodeGenerator.endl
        else:
            if force_name:
                param_name = force_name
            else:
                param_name = CodeGeneratorBase.get_expr_name(param)  # type:ignore
            if assign:
                param_str = CCodeGenerator._assign_variable(f"{prefix} {type_str} {param_name}", assign, op="=")
            else:
                param_str = f"{prefix} {type_str} {param_name}" + CCodeGenerator.endl
        return param_str.strip(" ")


    @staticmethod
    def _declare_parameter(param: Expr|Matrix, force_name: str = "", prefix: str = "") -> str:
        CodeGeneratorBase._raise_exception_non_variable(param)
        if force_name:
            param_name = force_name
        else:
            param_name = CodeGeneratorBase.get_expr_name(param)
        type_str = CCodeGenerator._get_type(param)
        param_str = f"{type_str} {param_name}"
        return param_str


    def instantiate_return_variable(self, expr: Expr|Matrix, name: str) -> str:
        ret = ""

        # TODO: this currently assumed flattened arrays

        ret = self._declare_variable(param=expr, force_name=name)

        # ret = return_type_str

        # # handle argument being an array
        # if self.is_array_type(expr):
        #     shape = param.shape  # type:ignore
        #     # shape_str = ", ".join([str(i) for i in shape])
        #     if len(shape) > 1:
        #         size = np.prod(shape)  # type:ignore
        #         if shape[1] == 1:
        #             constructor_str = "(" + str(size) + ")"
        #         else:
        #             constructor_str = "(" + str(size) + ")"
        #     else:
        #         constructor_str = "(" + str(shape[0]) + ")"
        #     param_str = f"std::vector<double> {name}" + constructor_str
        # else:
        #     param_str = f"double {name}"
        # ret += param_str + ";"

        #########################################

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


    def add_function(self,
                     expr: Expr|Matrix,
                     params: list,
                     function_name: str,
                     return_name: str,
                     use_cse: bool = True,
                     is_symmetric: bool = False,
                     description: str = ""):
        self.function_register.update({
            function_name: dict(
                expr=expr,
                params=params,
                return_name=return_name,
                use_cse=use_cse,
                is_symmetric=is_symmetric,
                description=description
                )
            })


    def subs_prune(self, replacements, expr_simples) -> tuple[dict, Matrix, int]:
        # unpack to single iterable
        reps = []
        for rep in replacements:
            if isinstance(rep, tuple) and (len(rep) == 2):
                reps += [rep]
            else:
                for r in rep:
                    reps += [r]

        # condense redundant replacements
        dreps = {sub: rexpr for sub, rexpr in reps}
        dreps_pops = []
        new_subs = {}  # new subs for expression to take out redundant variables

        # precompute and group subs by replacement expression
        repl_to_subs = defaultdict(list)
        # for sub, rexpr in dreps.items():
        for sub, rexpr in reps:
            repl_to_subs[rexpr].append(sub)

        # iterate over grouped expressions
        for sub, rexpr in tqdm(reps, ncols=100, desc="Pruning"):
            # if rexpr appears more than once in dict, its redundant
            if len(repl_to_subs[rexpr]) > 1:
                redundant_vars = repl_to_subs[rexpr]
                # replace redundant vars with first found var
                if redundant_vars:
                    keep_var = redundant_vars[0]
                    for rvar in redundant_vars[1:]:
                        new_subs.update({rvar: keep_var})
                        dreps_pops += [rvar]

        # original
        # for (sub, rexpr) in tqdm(dreps.items()):
        #     if rexpr in dreps.values():
        #         # get keys with this expression
        #         redundant_vars = [key for key, value in dreps.items() if value == rexpr]
        #         # replace redundant vars with first found var
        #         if redundant_vars:
        #             keep_var = redundant_vars[0]
        #             for rvar in redundant_vars[1:]:
        #                 new_subs.update({rvar: keep_var})
        #                 dreps_pops += [rvar]

        for var in dreps_pops:
            if var in dreps:
                dreps.pop(var)  # type:ignore

        #########################
        nchunk = 2_000
        remaining_chunk = [*dreps.items()]

        if len(remaining_chunk) > nchunk:
            chunked_dicts = []
            for i in range(0, len(remaining_chunk), nchunk):
                chunk = dict(remaining_chunk[i:i + nchunk])
                chunked_dicts += [chunk]

            chunked_new_subs = []
            nchunk_subs = 500
            new_subs_list = [*new_subs.items()]
            for i in range(0, len(new_subs), nchunk_subs):
                chunk_new_subs = dict(new_subs_list[i:i + nchunk_subs])
                chunked_new_subs += [chunk_new_subs]

            # remaining_chunk = dict(remaining_chunk[nchunk:])
            inter_reps = {}
            for chunk in tqdm(chunked_dicts, ncols=70, colour="yellow", desc="\tdict subs"):
                inter_reps.update(parallel_subs(chunk, chunked_new_subs))
            dreps = inter_reps
        else:
            dreps = parallel_subs(dict(remaining_chunk), [new_subs])
        # breakpoint()
        #########################

        # dreps = parallel_subs(dreps, [new_subs])
        replacements = [*dreps.items()]  # type:ignore

        expr_simples = parallel_subs(expr_simples, [new_subs])
        return (replacements, expr_simples, len(dreps_pops))  # type:ignore


    def _build_function(self, function_name: str, **func_register_info) -> list[str]:

        expr = func_register_info.get("expr")
        params = func_register_info.get("params")
        return_name = func_register_info.get("return_name")
        use_cse = func_register_info.get("use_cse")
        is_symmetric = func_register_info.get("is_symmetric")
        assert expr
        assert params
        assert return_name
        assert isinstance(use_cse, bool)
        assert isinstance(is_symmetric, bool)

        if use_cse:
            # old method
            # expr_replacements, expr_simple = cse(expr, symbols("X_temp0:1000"), optimizations='basic')

            # NOTE: handles MatrixSymbols in expression.
            # wrapping in Matrix() simplifies the form of
            # any matrix operation expressions.
            if expr.is_Matrix:
                expr = Matrix(expr)

            replacements, expr_simples = parallel_cse(expr)

            replacements, expr_simple, nredundant = self.subs_prune(replacements, expr_simples)

            for i in range(1, 10):
                replacements, expr_simple, nredundant = self.subs_prune(replacements, expr_simple)
                if nredundant == 0:
                    break

            # expr_replacements = tuple([(sub, repl) for sub, repl in dreps.items()])  # type:ignore
            expr_replacements = replacements

        else:
            expr_replacements = ()
            expr_simple = expr

        func_def = self.write_function_definition(name=function_name,
                                                  expr=expr_simple,
                                                  params=params)
        return_array = self.instantiate_return_variable(expr=expr, name=return_name)
        sub_expr = self.write_subexpressions(expr_replacements)
        func_body = self.write_matrix(matrix=Matrix(expr_simple),
                                      variable=return_name,
                                      is_symmetric=is_symmetric)
        func_ret = self.write_function_returns(expr=expr, return_names=[return_name])

        writes = [
                  func_def,
                  self.indent_lines(return_array),
                  self.indent_lines(sub_expr),
                  self.indent_lines(func_body),
                  self.indent_lines(func_ret),
                  "}"
                  ]

        return writes


    def create_module(self, module_name: str, path: str = ".", parallel: bool = False):
        # create extension module directory
        module_dir_name = module_name
        module_dir_path = os.path.join(path, module_dir_name)
        os.mkdir(module_dir_path)

        self.path = path
        self.file_name = module_name + ".cpp"
        file_path = os.path.join(module_dir_path, self.file_name)
        self.file = open(file_path, "w")

        header = self.write_header()
        self.write_lines(header)

        pybind_writes = ["", "", f"PYBIND11_MODULE({module_name}, m) " + "{"]  # }

        try:
            if parallel:
                # with Pool(processes=cpu_count()) as pool:
                #     pool_args = [(func_name, info) for func_name, info in self.function_register.items()]
                #     results = [pool.apply_async(self._build_function, arg) for arg in pool_args]
                #     results = [ret.get() for ret in results]
                pass
            else:
                # get functions from register
                for func_name, info in tqdm(self.function_register.items(), ncols=80, desc="Build"):
                    # build the function
                    writes = self._build_function(function_name=func_name, **info)
                    description = info["description"]
                    pybind_writes += [f"\tm.def(\"{func_name}\", &{func_name}, \"{description}\");"]
                    for line in writes:
                        self.write_lines(line)

                    # create __init__.py file
                    with open(os.path.join(module_dir_path, "__init__.py"), "a+") as f:
                        f.write(f"from {module_dir_name}.{module_dir_name} import {func_name}\n")

                pybind_writes += ["}"]

                for line in pybind_writes:
                    self.write_lines(line)

                self.create_build_file(path=module_dir_path,
                                       module_name=module_name,
                                       source=self.file_name)

                self.close()
        except Exception as e:
            shutil.rmtree(module_dir_path, ignore_errors=True)
            raise Exception(e)


    def create_build_file(self, module_name: str, path: str, source: str):
        # from pybind11.setup_helpers import Pybind11Extension
        # numpy_includes = ", ".join(np.get_include())

        file_name = source.split('.')[0]

        build_str = ("""\
        import os
        import sys
        import glob
        import shutil
        from setuptools import setup
        from setuptools.command.build_ext import build_ext
        from setuptools import Command
        from pybind11.setup_helpers import Pybind11Extension



        dir = os.path.dirname(__file__)

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
                    for file in glob.iglob(os.path.join(root_path, "**", pattern), recursive=True):
                        print("removing:", file)
                        os.remove(file)


        """f"""
        sources = [os.path.join(dir, "{source}")]

        # Define extension module
        ext_module = Pybind11Extension(name="{module_name}",
                                       sources=sources,
                                       extra_compile_args=['-std=c++14'],
                                       extra_link_args=['-std=c++14'])
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

        build_file_name = "build.py"
        build_file_path = os.path.join(path, build_file_name)
        with open(build_file_path, "a+") as f:
            f.write(dedent(build_str))

        # attempt to build
        # print(f"EXECUTING: python {build_file_path}")
        # os.system(f"python {build_file_path}")


    def close(self):
        self.file.close()
