import os
import shutil
from typing import Optional
from tqdm import tqdm
import numpy as np
from japl.BuildTools.CodeGeneratorBase import CodeGeneratorBase
from sympy import ccode
from sympy import Expr
from sympy import Matrix
from sympy import Symbol
from sympy import cse
from sympy import Piecewise
from sympy.codegen.ast import float64, real
from textwrap import dedent
from japl.BuildTools.BuildTools import parallel_subs
from japl.BuildTools.BuildTools import parallel_cse
from collections import defaultdict
from japl.Symbolic.KwargFunction import KwargFunction



class CCodeGenerator(CodeGeneratorBase):

    __JAPL_EXT_MODULE_INIT_HEADER = "# __JAPL_EXTENSION_MODULE__\n"
    comment_prefix: str = "//"
    pre_bracket: str = "["
    post_bracket: str = "]"
    bracket_separator: str = "]["  # ]
    endl: str = ";\n"

    header: list[str] = ["#include <iostream>",
                         "#include <model.hpp>",
                         "#include <pybind11/pybind11.h>",
                         "#include <pybind11/numpy.h>",
                         "#include <pybind11/stl.h>  // Enables automatic conversion",
                         "",
                         "namespace py = pybind11;",
                         ""]

    def __init__(self, strict: bool = False, use_std_args: bool = False):
        self.strict = strict
        self.function_register = {}
        self.use_std_args = use_std_args  # use standard Model args


    def _get_code(self, expression):
        return ccode(expression, type_aliases={real: float64}, strict=self.strict)


    def _handle_peicewise_recursive(self, expr: Expr) -> Expr:
        def pw(expr) -> Optional[dict]:
            for arg in expr.args:
                if arg.has(Piecewise):
                    return pw(arg)
            if isinstance(expr, Piecewise):
                return {expr: Symbol(self._get_code(expr))}

        if expr.has(Piecewise):
            ret = pw(expr)
            if ret is not None:
                expr = expr.subs(ret)  # type:ignore
        return expr


    def _write_subexpressions(self, subexpressions: list|dict) -> str:
        if isinstance(subexpressions, dict):
            subexpressions = list(subexpressions.items())
        ret = ""
        for (lvalue, rvalue) in subexpressions:
            rvalue = self._handle_peicewise_recursive(rvalue)  # handle Piecewise recursively
            assign_str = self._declare_variable(lvalue, prefix="const", assign=self._get_code(rvalue))  # type:ignore
            ret += assign_str
        return ret


    def _write_matrix(self,
                      matrix,
                      variable,
                      is_symmetric=False):
        write_string = ""
        variable_name = self._get_expr_name(variable)

        ############################
        # L-value / R-value assign
        ############################
        # if Matrix of single expression
        if matrix.shape[0] * matrix.shape[1] == 1:
            write_string = write_string + variable_name + " = " + self._get_code(matrix[0]) + self.endl  # type:ignore
        # if row or column Matrix
        elif matrix.shape[0] == 1 or matrix.shape[1] == 1:
            for i in range(0, len(matrix)):
                write_string = write_string + variable_name +\
                               self.pre_bracket +\
                               str(i) +\
                               self.post_bracket + " = " +\
                               self._get_code(matrix[i]) + self.endl  # type:ignore
        # if any other shape
        else:
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
                                    + self._get_code(matrix[i, j]) + self.endl)  # type:ignore

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


    def _write_function_definition(self, name: str, expr: Expr|Matrix, params: list,
                                   by_reference: dict = {}):
        return_type_str = self._get_return_type(expr)
        params_list, params_unpack = self._get_function_parameters(params=params,
                                                                   by_reference=by_reference)
        func_proto = f"{return_type_str} {name}({params_list})" + " {\n"  # }
        return func_proto + self._indent_lines(params_unpack)


    def _write_function_returns(self, expr: Expr|Matrix, return_names: list[str]):
        if len(return_names) > 1:
            raise Exception("CCodeGenerator currently only supports returns of a"
                            "single object.")
        return_name = return_names[0]
        if self._is_array_type(expr):
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
        if CodeGeneratorBase._is_array_type(param):
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

        if CodeGeneratorBase._is_array_type(param):
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
                param_name = CodeGeneratorBase._get_expr_name(param)
            if assign:
                param_str = CCodeGenerator._assign_variable(f"{prefix} {type_str} {param_name}", assign, op="=")
            else:
                param_str = f"{prefix} {type_str} {param_name}" + CCodeGenerator.endl
        return param_str.strip(" ")


    @staticmethod
    def _declare_parameter(param: Expr|Matrix, force_name: str = "", by_reference: dict = {}) -> str:
        CodeGeneratorBase._raise_exception_non_variable(param)
        if force_name:
            param_name = force_name
        else:
            param_name = CodeGeneratorBase._get_expr_name(param)
        type_str = CCodeGenerator._get_type(param)

        # TODO: probably move this to _get_type
        # handle pass-by-reference
        # NOTE: currently does not work for array types
        ref_char = ''
        if CodeGeneratorBase._is_array_type(param):
            pass
        #     # check if all items in param are defined
        #     # in by_reference. if so, this param can be
        #     # passed by reference.
        #     array_check = [p in by_reference for p in param]  # type:ignore
        #     if np.asarray(array_check).all():
        #         # ref_char = '&'
        #         # use this type when passing array type by
        #         # reference
        #         type_str = "py::array"
        else:
            if param in by_reference:
                ref_char = '&'

        param_str = f"{type_str} {ref_char}{param_name}"
        return param_str


    def _instantiate_return_variable(self, expr: Expr|Matrix, name: str) -> str:
        # TODO: this currently assumed flattened arrays
        ret = self._declare_variable(param=expr, force_name=name)
        return ret


    def add_function(self,
                     expr: Expr|Matrix,
                     params: list,
                     function_name: str,
                     return_name: str,
                     use_cse: bool = True,
                     is_symmetric: bool = False,
                     description: str = "",
                     by_reference: dict = {}):
        # convert keys of type string in reference info
        # to Symbols
        pops = []
        adds = []
        for key, val in by_reference.items():
            if isinstance(key, str):
                adds += [(Symbol(key), val)]
                pops += [key]
        for param, val in adds:
            by_reference[param] = val
        for key in pops:
            by_reference.pop(key)

        # register function info
        self.function_register.update({
            function_name: dict(
                expr=expr,
                params=params,
                return_name=return_name,
                use_cse=use_cse,
                is_symmetric=is_symmetric,
                description=description,
                by_reference=by_reference,
                )
            })


    @staticmethod
    def _subs_prune(replacements, expr_simple) -> tuple[dict, Matrix, int]:
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
            for chunk in tqdm(chunked_dicts, ncols=70, desc="\tdict subs",
                              ascii=" ="):
                inter_reps.update(parallel_subs(chunk, chunked_new_subs))  # type:ignore
            dreps = inter_reps
        else:
            dreps = parallel_subs(dict(remaining_chunk), [new_subs])

        replacements = [*dreps.items()]  # type:ignore

        expr_simple = parallel_subs(expr_simple, [new_subs])
        return (replacements, expr_simple, len(dreps_pops))  # type:ignore


    def _build_function(self, function_name: str, **func_register_info) -> list[str]:

        MAX_PRUNE_ITER = 10

        expr = func_register_info.get("expr")
        params = func_register_info.get("params")
        return_name = func_register_info.get("return_name")
        use_cse = func_register_info.get("use_cse")
        is_symmetric = func_register_info.get("is_symmetric")
        by_reference = func_register_info.get("by_reference", {})
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

            ######################################################
            # optimize pass-by-reference exprs
            ######################################################
            # add by_reference expressions to expr and
            # do cse optimization to get substitutions.
            # then split main expr & pass-by-reference
            # expression again for individual processing
            by_ref_nadds = len(by_reference)
            if by_ref_nadds > 0:
                by_ref_expr = Matrix([*by_reference.values()])
                expr = Matrix([*expr, *by_ref_expr])
            ######################################################

            replacements, expr_simple = cse(expr)
            expr_simple = expr_simple[0]

            # must further optimize and make substitutions
            # between indices of expr
            for _ in range(MAX_PRUNE_ITER):
                replacements, expr_simple, nredundant = self._subs_prune(replacements, expr_simple)
                if nredundant == 0:
                    break

            ######################################################
            # optimize pass-by-reference exprs
            ######################################################
            if by_ref_nadds > 0:
                expr_simple = expr_simple[:-by_ref_nadds]
                for i, (k, v) in enumerate(by_reference.items()):
                    by_reference[k] = expr_simple[-by_ref_nadds:][i]
            ######################################################

            # remove added reference expr which were
            # added for cse()
            if by_ref_nadds > 0:
                expr = Matrix(expr[:-by_ref_nadds])

        else:
            replacements = []
            expr_simple = expr

        return_type_str = self._get_return_type(expr_simple)
        params_list, params_unpack = self._get_function_parameters(params=params,
                                                                   by_reference=by_reference,
                                                                   use_std_args=self.use_std_args)
        func_proto = f"{return_type_str} {function_name}({params_list})" + " {\n"  # }
        func_def = func_proto + self._indent_lines(params_unpack)
        # old
        # func_def = self._write_function_definition(name=function_name,
        #                                            expr=expr_simple,
        #                                            params=params,
        #                                            by_reference=by_reference)

        return_array = self._instantiate_return_variable(expr=expr, name=return_name)
        sub_expr = self._write_subexpressions(replacements)  # type:ignore
        func_body = self._write_matrix(matrix=Matrix(expr_simple),
                                       variable=return_name,
                                       is_symmetric=is_symmetric)
        func_ret = self._write_function_returns(expr=expr, return_names=[return_name])

        writes = [
                  func_def,
                  self._indent_lines(return_array),
                  self._indent_lines(sub_expr),
                  self._indent_lines(func_body),
                  ]

        #######################################
        # write other defined subexpressions
        #######################################
        # find '&' (pass-by-reference) params in parameter name list
        by_ref_params_list = [i for i in params_list.split(",") if ('&' in i) or ("py::array" in i)]
        for ref_param in by_ref_params_list:
            if '&' in ref_param:
                ref_param_type, ref_param_name = ref_param.split('&')
                # TODO this does nothing
            elif "py::array" in ref_param:
                ref_param_type, ref_param_name = ref_param.split('py::array')

                #####
                ref_param_type = ref_param_type.strip()
                ref_param_name = ref_param_name.strip()

                # add line which gets pointer to data from py::array type
                by_ref_param_name = ref_param_name

                by_ref_size_check_str = (f"if ({by_ref_param_name}.size() != {len(by_reference)})"
                                         " {\n"  # }
                                         f"\tthrow std::length_error(\""  # )
                                         f"expected length of {len(by_reference)} for argument"
                                         f"{by_ref_param_name} but instead got length \" "
                                         f"+ std::to_string({by_ref_param_name}.size())){self.endl}"
                                         "}\n"
                                         )

                by_ref_buf_name = f"{by_ref_param_name}_buf"
                by_ref_buf_str = f"py::buffer_info {by_ref_buf_name} = {by_ref_param_name}.request()" + self.endl

                by_ref_ptr_name = f"{by_ref_param_name}_ptr"
                by_ref_ptr_str = (f"double* {by_ref_ptr_name} = static_cast<double*>"
                                  f"({by_ref_buf_name}.ptr)" + self.endl)
                by_ref_dtype = f"{by_ref_buf_name}.format"

                by_ref_check_dtype_str = (f"if ({by_ref_dtype} != \"d\") "
                                          "{\n"
                                          f"\tthrow py::type_error(\"attempting to pass argument "  # )
                                          f"{by_ref_param_name}, but "
                                          f"must be of type double\"){self.endl}"
                                          "}\n")
                by_ref_ptr_str = by_ref_size_check_str + by_ref_buf_str + by_ref_ptr_str + by_ref_check_dtype_str
                #####
                # NOTE: this is an augmented _write_subexpressions()
                # allowing the specification of:
                #   - lvalue variable name
                #   - lvalue variable being array-type and accessed
                #   - type being pointer
                by_ref_subexpr_str = ""
                for i, (lvalue, rvalue) in enumerate(by_reference.items()):  # type:ignore
                    # type_str = self._get_type(lvalue)
                    accessor_str = self.pre_bracket + str(i) + self.post_bracket
                    lvalue_str = f"{by_ref_ptr_name}{accessor_str}"
                    assign_str = self._assign_variable(lvalue_str, self._get_code(rvalue))  # type:ignore
                    by_ref_subexpr_str += assign_str

                # usr_sub_expr_simple = parallel_subs(by_reference, replacements)
                writes += [self._indent_lines(by_ref_ptr_str),
                           self._indent_lines(by_ref_subexpr_str)]

            else:
                raise Exception("unhandled case, but also make this code better.")

            ###################################################
            # by_ref_param_name = ref_param_name
            # by_ref_ptr_name = f"{by_ref_param_name}_ptr"
            # by_ref_ptr_str = (f"double* {by_ref_ptr_name} = static_cast<double*>"
            #                   f"({by_ref_param_name}.mutable_data())" + self.endl)
            ###################################################

            # apply subs from previous code to subexpressions
            # breakpoint()
            # by_ref_replacements, by_ref_expr_simple = parallel_cse(Matrix([*by_reference.values()]))
            # breakpoint()
            # by_reference = parallel_subs(by_reference, [replacements])  # type:ignore
            # for _ in range(MAX_PRUNE_ITER):
            #     replacements, expr_simple, nredundant = self._subs_prune(replacements, Matrix([*by_reference.values()]))
            #     if nredundant == 0:
            #         break


        # close function
        writes += [self._indent_lines(func_ret),
                   "}"]

        return writes


    def create_module(self, module_name: str, path: str = "."):
        # create extension module directory
        module_dir_name = module_name
        module_dir_path = os.path.join(path, module_dir_name)
        os.mkdir(module_dir_path)

        # output directory warning
        # if os.path.isdir(module_dir_path):
        #     raise Exception(f"output directory {module_dir_name} already exists")

        self.path = path
        self.file_name = module_name + ".cpp"
        file_path = os.path.join(module_dir_path, self.file_name)
        self.file = open(file_path, "w")

        header = self._write_header()
        self._write_lines(header)

        pybind_writes = ["", "", f"PYBIND11_MODULE({module_name}, m) " + "{"]  # }

        try:
            # get functions from register
            for func_name, info in tqdm(self.function_register.items(), ncols=80, desc="Build"):
                # handle func_name references class method "Class::method"
                class_ref = ""
                if "::" in func_name:
                    _func_str_split = func_name.split("::")
                    class_ref = "".join(_func_str_split[0])
                    func_name = "".join(_func_str_split[1:])
                # build the function
                writes = self._build_function(function_name=func_name, **info)
                description = info["description"]
                pybind_writes += [f"\tm.def(\"{func_name}\", &{class_ref}::{func_name}, \"{description}\");"]
                for line in writes:
                    self._write_lines(line)

                # create __init__.py file
                with open(os.path.join(module_dir_path, "__init__.py"), "a+") as f:
                    f.write(self.__JAPL_EXT_MODULE_INIT_HEADER)
                    f.write(f"from {module_dir_name}.{module_dir_name} import {func_name}\n")

            pybind_writes += ["}"]

            for line in pybind_writes:
                self._write_lines(line)

            self.create_build_file(path=module_dir_path,
                                   module_name=module_name,
                                   source=self.file_name)

            self._close()
        except Exception as e:
            shutil.rmtree(module_dir_path, ignore_errors=True)
            raise Exception(e)


    def create_build_file(self, module_name: str, path: str, source: str):
        file_name = source.split('.')[0]
        cxx_std = 17

        build_str = ("""\
        import os
        import sys
        import glob
        import shutil
        from setuptools import setup
        from setuptools.command.build_ext import build_ext
        from setuptools import Command
        from pybind11.setup_helpers import Pybind11Extension
        from japl.global_opts import JAPL_HOME_DIR



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
                                       extra_compile_args=[],
                                       extra_link_args=[],
                                       include_dirs=[os.path.join(JAPL_HOME_DIR, "libs", "model")],
                                       cxx_std={cxx_std})
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


    def _close(self):
        self.file.close()
