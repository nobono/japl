import os
import sys
import shutil
from tqdm import tqdm
import numpy as np
from japl.BuildTools.CodeGeneratorBase import CodeGeneratorBase
from sympy import ccode
from sympy import Expr
from sympy import Matrix
from sympy import symbols
from sympy import cse
from sympy import simplify
from sympy.codegen.ast import float64, real
from textwrap import dedent
from multiprocess import Pool  # type:ignore
from multiprocess import cpu_count  # type:ignore
import dill as pickle
from japl.BuildTools.BuildTools import parallel_subs
from japl.BuildTools.BuildTools import dict_subs_func
from japl.BuildTools.BuildTools import parallel_cse
from multiprocess import Pool  # type:ignore
from multiprocess import cpu_count  # type:ignore



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

        # TODO: this currently assumed flattened arrays

        # handle argument being an array
        if self.is_array_type(param):
            shape = param.shape  # type:ignore
            # shape_str = ", ".join([str(i) for i in shape])
            if len(shape) > 1:
                size = np.prod(shape)  # type:ignore
                if shape[1] == 1:
                    constructor_str = "(" + str(size) + ")"
                else:
                    constructor_str = "(" + str(size) + ")"
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
            # expr_replacements, expr_simple = cse(expr, symbols("X_temp0:1000"), optimizations='basic')

            replacements, expr_simples = parallel_cse(expr)


            # def redundancy(pair, rexpr):
            #     sub, expr = pickle.loads(pair)
            #     rexpr = pickle.loads(rexpr)
            #     return pickle.dumps(sub if expr == rexpr else None)


            def redundancy(sub, rexpr, dreps):
                sub = pickle.loads(sub)
                rexpr = pickle.loads(rexpr)
                dreps = pickle.loads(dreps)

                dreps_pops = []
                new_subs = {}  # new subs for expression to take out redundant variables
                # for (sub, rexpr) in tqdm(dreps.items()):
                    # print("redundancy iter:", i, "of", dreps_len)
                    # dreps_exprs = list(dreps.values())
                    # dreps_keys = list(dreps.keys())

                if rexpr in dreps.values():
                    # get keys with this expression
                    redundant_vars = [key for key, value in dreps.items() if value == rexpr]

                    # replace redundant vars with first found var
                    if redundant_vars:
                        keep_var = redundant_vars[0]
                        for rvar in redundant_vars[1:]:
                            new_subs.update({rvar: keep_var})
                            dreps_pops += [rvar]
                return pickle.dumps((dreps_pops, new_subs))


            def subs_prune(replacements, expr_simples) -> tuple[dict, Matrix, int]:
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

                # dreps_len = len(dreps)
                for (sub, rexpr) in tqdm(dreps.items()):
                    # print("redundancy iter:", i, "of", dreps_len)
                    # dreps_exprs = list(dreps.values())
                    # dreps_keys = list(dreps.keys())

                    if rexpr in dreps.values():
                        # get keys with this expression
                        redundant_vars = [key for key, value in dreps.items() if value == rexpr]

                        # replace redundant vars with first found var
                        if redundant_vars:
                            keep_var = redundant_vars[0]
                            for rvar in redundant_vars[1:]:
                                new_subs.update({rvar: keep_var})
                                dreps_pops += [rvar]

                ##################
                # with Pool(processes=cpu_count()) as pool:
                #     args = [(pickle.dumps(sub), pickle.dumps(rexpr), pickle.dumps(dreps)) for sub, rexpr in dreps.items()]
                #     results = [pool.apply_async(redundancy, arg) for arg in args]
                #     results = [pickle.loads(ret.get()) for ret in results]
                # dreps_pops = [r for ret in results for r in ret[0]]
                # new_subs_res = [r[1] for r in results]
                # for d in new_subs_res:
                #     new_subs.update(d)
                ##################

                dreps = parallel_subs(dreps, [new_subs])
                for var in dreps_pops:
                    if var in dreps:
                        dreps.pop(var)  # type:ignore
                expr_simples = parallel_subs(expr_simples, [new_subs])
                return (dreps, expr_simples, len(dreps_pops))  # type:ignore


            dreps, expr_simple, nredundant = subs_prune(replacements, expr_simples)

            for i in range(1, 10):
                replacements = tuple([(k, v) for k, v in dreps.items()])
                dreps, expr_simple, nredundant = subs_prune(replacements, expr_simple)
                print("num redundant:", nredundant)
                if nredundant == 0:
                    break
                print("prune iter:", i)

            expr_replacements = tuple([(sub, repl) for sub, repl in dreps.items()])  # type:ignore

        else:
            expr_replacements = ()
            expr_simple = expr

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

        return writes


    def create_module(self, module_name: str, path: str, parallel: bool = False):
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
                for func_name, info in self.function_register.items():
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
