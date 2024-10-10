import re
from io import TextIOWrapper
from sympy import ccode, octave_code, pycode
from sympy import Expr
from sympy import Matrix
from sympy import MatrixSymbol
from sympy import Symbol
from sympy import Function
from sympy import symbols
from sympy import cse
from sympy.codegen.ast import float32, real
from sympy.matrices.expressions.matexpr import MatrixElement
from japl.Util.Util import flatten_list



class CodeGeneratorBase:

    file_name: str
    file: TextIOWrapper
    comment_prefix: str

    pre_bracket: str
    post_bracket: str
    bracket_separator: str
    endl: str

    header: list[str] = []
    footer: list[str] = []

    def print_string(self, string: str) -> None:
        self.file.write(f"{self.comment_prefix} " + string + "\n")


    @staticmethod
    def indent_lines(string: str) -> str:
        ret = ""
        for line in string.split("\n"):
            ret += "\t" + line + "\n"
        return ret


    def write_lines(self, string, prefix: str = "", postfix: str = "\n"):
        for line in string.split('\n'):
            self.file.write(prefix + line + postfix)


    def write_function_definition(self, *args, **kwargs) -> str:
        return ""


    def write_function_returns(self, expr: Expr, return_names: list[str]) -> str:
        return ""


    def write_subexpressions(self, *args, **kwargs) -> str:
        return ""


    def write_matrix(self, *args, **kwargs) -> str:
        return ""


    def declare_parameter(self, param: Expr, name: str = "", *args, **kwargs) -> str:
        if name:
            return name
        else:
            return self.get_expr_name(param)


    def close(self):
        pass


    def write_header(self) -> str:
        # for line in self.header:
        #     self.write_lines(line)
        return "\n".join(self.header)


    def write_footer(self) -> str:
        # for line in self.footer:
        #     self.write_lines(line)
        return "\n".join(self.footer)


    def write_function_to_file(self, path: str, function_name: str, expr: Expr,
                               input_vars: list|tuple, return_var: Expr|str,
                               is_symmetric: bool = False):
        self.file_name = path
        self.file = open(self.file_name, 'w')
        expr_replacements, expr_simple = cse(expr, symbols("X_temp0:1000"), optimizations='basic')

        if isinstance(return_var, str):
            if hasattr(expr, "shape"):
                return_var = MatrixSymbol(return_var, *expr.shape)  # type:ignore
            else:
                return_var = Symbol(return_var)

        self.write_header()
        self.print_string("Equations for state matrix prediction")
        self.write_function_definition(name=function_name,
                                       params=input_vars,
                                       returns=[return_var])
        self.write_subexpressions(expr_replacements)
        self.write_matrix(matrix=Matrix(expr_simple),
                          variable=return_var,
                          is_symmetric=is_symmetric)
        self.write_function_returns(expr=expr, return_names=[return_var])
        self.write_lines("")
        self.write_footer()
        self.close()


    def is_array_type(self, param: Expr|Matrix) -> bool:
        if hasattr(param, "__len__") or hasattr(param, "shape"):
            return True
        else:
            return False


    def get_function_parameters(self, params: list[Expr]) -> tuple[str, str]:
        """returns names of paramters as a string. If Matrix or list of
        parameters provided as one of the parameters, return the string
        which unpacks the dummy variables."""
        dummy_prefix = "_Dummy_var"
        arg_names = []
        arg_unpack_str = ""
        for i, param in enumerate(params):
            if self.is_array_type(param):
                # unpack iterable param
                dummy_name = dummy_prefix + str(i)
                for ip, p in enumerate(param):  # type:ignore
                    # if p is MatrixElement no need to use
                    # a dummy var, allow accessing parameter
                    # directly
                    if isinstance(p, MatrixElement):
                        dummy_name = p.parent.name
                    else:
                        unpack_var = self.declare_parameter(p)
                        accessor_str = self.pre_bracket + str(ip) + self.post_bracket
                        arg_unpack_str += f"{unpack_var} = {dummy_name}{accessor_str}" + self.endl
                # store dummy var in arg_names
                arg_names += [self.declare_parameter(param, name=dummy_name)]
            else:
                # if single symbol
                param_str = self.declare_parameter(param)
                arg_names += [param_str]
        arg_names_str = ", ".join(arg_names)
        return (arg_names_str, arg_unpack_str)


    @staticmethod
    def get_expr_name(expr: Expr) -> str:
        if hasattr(expr, "name"):
            return getattr(expr, "name")
        else:
            return str(expr)


class OctaveCodeGenerator(CodeGeneratorBase):

    comment_prefix: str = "%"
    pre_bracket: str = "("
    post_bracket: str = ")"
    bracket_separator: str = ", "
    endl: str = ";\n"

    def __init__(self):
        # self.file_name = file_name
        # self.file = open(self.file_name, 'w')
        pass


    def get_code(self, expression):
        # return ccode(expression, type_aliases={real:float32})
        return octave_code(expression)


    def write_subexpressions(self, subexpressions):
        write_string = ""
        for (lvalue, rvalue) in subexpressions:
            write_string = write_string + str(lvalue) + " = " + self.get_code(rvalue) + self.endl  # type:ignore

        write_string = self.transform_to_octave_style(write_string)
        self.write_lines(write_string, prefix="\t")


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
                               self.pre_bracket + str(i + 1) +\
                               self.post_bracket + " = " +\
                               self.get_code(matrix[i]) + self.endl  # type:ignore
                write_string = self.transform_to_octave_style(write_string)
        # if any other shape
        else:
            for j in range(0, matrix.shape[1]):
                for i in range(0, matrix.shape[0]):
                    if j >= i or not is_symmetric:
                        write_string = write_string + variable_name +\
                                self.pre_bracket +\
                                str(i + 1) + self.bracket_separator + str(j + 1) +\
                                self.post_bracket + " = " +\
                                self.get_code(matrix[i, j]) + self.endl  # type:ignore
                        write_string = self.transform_to_octave_style(write_string)

        self.write_lines(write_string, prefix="\t")


    def transform_to_octave_style(self, input_string):
        # Define the pattern to match: word[index][index]
        pattern = r'(\w+)\[(\w+)\]\[(\w+)\]'

        # Define the replacement function
        def replacer(match):
            var_name = match.group(1)  # The variable name (e.g., var)
            try:
                i = int(match.group(2)) + 1  # First index, converted to integer and incremented by 1
            except Exception as _:  # noqa
                i = str(match.group(2)) + '+1'

            try:
                j = int(match.group(3)) + 1  # Second index, converted to integer and incremented by 1
            except Exception as _:  # noqa
                j = str(match.group(3)) + '+1'
            return f"{var_name}({i}, {j})"

        # Apply the substitution
        result = re.sub(pattern, replacer, input_string)
        return result


    def write_function_definition(self, name, params, returns):
        params_list, params_unpack = self.get_function_parameters(params)
        return_names = ", ".join(str(i) for i in returns)
        string = f"function [{return_names}] = {name}({params_list})\n\n"
        self.file.write(string)
        # write unpacked paramters
        self.write_lines(params_unpack, prefix="\t")


    def close(self):
        self.file.write("end\n")
        self.file.close()


class PyCodeGenerator(CodeGeneratorBase):

    comment_prefix: str = "#"
    pre_bracket: str = "["
    post_bracket: str = "]"
    bracket_separator: str = ']['  # ]
    endl: str = "\n"

    header: list[str] = ["import numpy as np", "", "", ""]

    def __init__(self):
        # self.file_name = file_name
        # self.file = open(self.file_name, 'w')
        pass


    def get_code(self, expression):
        return pycode(expression)


    def write_subexpressions(self, subexpressions):
        write_string = ""
        for (lvalue, rvalue) in subexpressions:
            write_string = write_string + str(lvalue) + " = " + self.get_code(rvalue) + self.endl  # type:ignore
        self.write_lines(write_string, prefix="\t")


    def write_matrix(self,
                     matrix,
                     variable,
                     is_symmetric=False):
        write_string = ""
        variable_name = self.get_expr_name(variable)

        # declare return var
        self.instantiate_return_array(variable)

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
                               str(i + 1) +\
                               self.post_bracket + " = " +\
                               self.get_code(matrix[i]) + self.endl  # type:ignore
        # if any other shape
        else:
            for j in range(0, matrix.shape[1]):
                for i in range(0, matrix.shape[0]):
                    if j >= i or not is_symmetric:
                        write_string = write_string + variable_name +\
                                       self.pre_bracket +\
                                       str(i + 1) + self.bracket_separator + str(j + 1) +\
                                       self.post_bracket + " = " +\
                                       self.get_code(matrix[i, j]) + self.endl  # type:ignore

        self.write_lines(write_string, prefix="\t")


    def write_function_definition(self, name, params, **kwargs):
        params_list, params_unpack = self.get_function_parameters(params)
        string = f"def {name}({params_list}):\n\n"
        self.file.write(string)
        # write unpacked paramters
        self.write_lines(params_unpack, prefix="\t")


    def write_function_returns(self, returns):
        return_names, params_unpack = self.get_function_parameters(returns)
        self.file.write(f"\treturn ({return_names})")


    def instantiate_return_array(self, param: Expr):
        return_name = self.get_expr_name(param)

        # handle argument being an array
        if hasattr(param, "__len__") or hasattr(param, "shape"):
            shape_str = ", ".join([str(i) for i in param.shape])  # type:ignore
            param_str = f"{return_name} = np.zeros(({shape_str}))"
        else:
            param_str = f"double {return_name}"
        self.write_lines(param_str, prefix="\t")


    def close(self):
        self.file.close()
