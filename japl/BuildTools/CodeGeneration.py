import re
from io import TextIOWrapper
from sympy import ccode, octave_code
from sympy import Expr
from sympy import Matrix
from sympy import symbols
from sympy import cse
from sympy.codegen.ast import float32, real



class CodeGeneratorBase:

    file_name: str
    file: TextIOWrapper
    comment_prefix: str

    def print_string(self, string: str) -> None:
        self.file.write(f"{self.comment_prefix} " + string + "\n")


    def write_lines(self, string, prefix: str = "", postfix: str = "\n"):
        for line in string.split('\n'):
            self.file.write(prefix + line + postfix)


    def write_function_definition(self, *args, **kwargs):
        pass


    def write_subexpressions(self, *args, **kwargs):
        pass


    def write_matrix(self, *args, **kwargs):
        pass


    def close(self):
        pass


    def write_function_to_file(self, path: str, function_name: str, expr: Expr,
                               input_vars: list|tuple, return_var: str,
                               is_symmetric: bool = False):
        self.file_name = path
        self.file = open(self.file_name, 'w')
        expr_simple = cse(expr, symbols("X_temp0:1000"), optimizations='basic')
        # cov_code_generator = OctaveCodeGenerator(JAPL_HOME_DIR
        #                                          + "/derivation/nav/generated/"
        #                                          + "state_predict.m")
        self.print_string("Equations for state matrix prediction")
        self.write_function_definition(name=function_name,
                                       args=input_vars,
                                       returns=[return_var])
        self.write_subexpressions(expr_simple[0])
        self.write_matrix(matrix=Matrix(expr_simple[1]),
                          variable_name=return_var,
                          is_symmetric=is_symmetric)
        # cov_code_generator.write_function_returns(returns=return_args)
        self.close()


class OctaveCodeGenerator(CodeGeneratorBase):

    comment_prefix: str = "%"

    def __init__(self):
        # self.file_name = file_name
        # self.file = open(self.file_name, 'w')
        pass


    def get_code(self, expression):
        # return ccode(expression, type_aliases={real:float32})
        return octave_code(expression)


    def write_subexpressions(self, subexpressions):
        write_string = ""
        for item in subexpressions:
            write_string = write_string + str(item[0]) + " = " + self.get_code(item[1]) + ";\n"  # type:ignore

        write_string = self.transform_to_octave_style(write_string)
        self.write_lines(write_string, prefix="\t")


    def write_matrix(self,
                     matrix,
                     variable_name,
                     is_symmetric=False,
                     pre_bracket="(",
                     post_bracket=")",
                     separator=", "):
        write_string = ""
        if matrix.shape[0] * matrix.shape[1] == 1:
            write_string = write_string + variable_name + " = " + self.get_code(matrix[0]) + ";\n"
        elif matrix.shape[0] == 1 or matrix.shape[1] == 1:
            for i in range(0, len(matrix)):
                write_string = write_string + variable_name +\
                               pre_bracket + str(i + 1) +\
                               post_bracket + " = " +\
                               self.get_code(matrix[i]) + ";\n"
                write_string = self.transform_to_octave_style(write_string)
        else:
            for j in range(0, matrix.shape[1]):
                for i in range(0, matrix.shape[0]):
                    if j >= i or not is_symmetric:
                        write_string = write_string + variable_name +\
                                pre_bracket +\
                                str(i + 1) + separator + str(j + 1) +\
                                post_bracket + " = " +\
                                self.get_code(matrix[i, j]) + ";\n"
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


    def write_function_definition(self, name, args, returns):
        arg_names = ", ".join(str(i) for i in args)
        return_names = ", ".join(str(i) for i in returns)
        string = f"function [{return_names}] = {name}({arg_names})\n\n"
        self.file.write(string)


    def close(self):
        self.file.write("end\n")
        self.file.close()


class CCodeGenerator(CodeGeneratorBase):

    comment_prefix: str = "//"

    def __init__(self, strict: bool = False):
        self.strict = strict
        # self.file_name = file_name
        # self.file = open(self.file_name, 'w')


    def get_code(self, expression):
        return ccode(expression, type_aliases={real: float32}, strict=self.strict)


    def write_subexpressions(self, subexpressions):
        write_string = ""
        for item in subexpressions:
            write_string = write_string + str(item[0]) + " = " + self.get_code(item[1]) + ";\n"  # type:ignore

        write_string = write_string + "\n\n"
        # self.file.write(write_string)
        self.write_lines(write_string, prefix="\t")


    def write_matrix(self,
                     matrix,
                     variable_name,
                     is_symmetric=False,
                     pre_bracket="[",
                     post_bracket="]",
                     separator="]["):
        write_string = ""
        if matrix.shape[0] * matrix.shape[1] == 1:
            write_string = write_string + variable_name + " = " + self.get_code(matrix[0]) + ";\n"
        elif matrix.shape[0] == 1 or matrix.shape[1] == 1:
            for i in range(0, len(matrix)):
                write_string = write_string + variable_name +\
                               pre_bracket +\
                               str(i + 1) +\
                               post_bracket + " = " +\
                               self.get_code(matrix[i]) + ";\n"

        else:
            for j in range(0, matrix.shape[1]):
                for i in range(0, matrix.shape[0]):
                    if j >= i or not is_symmetric:
                        write_string = write_string + variable_name +\
                                       pre_bracket +\
                                       str(i + 1) + separator + str(j + 1) +\
                                       post_bracket + " = " +\
                                       self.get_code(matrix[i, j]) + ";\n"

        write_string = write_string + "\n\n"
        self.write_lines(write_string, prefix="\t")
        self.write_lines("}")


    def write_function_definition(self, name, args):
        arg_names = ", ".join(str(i) for i in args)
        string = f"void {name}({arg_names})" + " {\n\n"
        self.file.write(string)


    def close(self):
        self.file.close()



class PyCodeGenerator(CodeGeneratorBase):

    comment_prefix: str = "#"

    def __init__(self, file_name):
        # self.file_name = file_name
        # self.file = open(self.file_name, 'w'
        pass


    def get_code(self, expression):
        return ccode(expression, type_aliases={real: float32})


    def write_subexpressions(self, subexpressions):
        write_string = ""
        for item in subexpressions:
            write_string = write_string + str(item[0]) + " = " + self.get_code(item[1]) + "\n"  # type:ignore

        write_string = write_string + "\n\n"
        # self.file.write(write_string)
        self.write_lines(write_string, prefix="\t")


    def write_matrix(self,
                     matrix,
                     variable_name,
                     is_symmetric=False,
                     pre_bracket="[",
                     post_bracket="]",
                     separator="]["):
        write_string = ""
        if matrix.shape[0] * matrix.shape[1] == 1:
            write_string = write_string + variable_name + " = " + self.get_code(matrix[0]) + "\n"
        elif matrix.shape[0] == 1 or matrix.shape[1] == 1:
            for i in range(0, len(matrix)):
                write_string = write_string + variable_name +\
                               pre_bracket +\
                               str(i + 1) +\
                               post_bracket + " = " +\
                               self.get_code(matrix[i]) + "\n"

        else:
            for j in range(0, matrix.shape[1]):
                for i in range(0, matrix.shape[0]):
                    if j >= i or not is_symmetric:
                        write_string = write_string + variable_name +\
                                       pre_bracket +\
                                       str(i + 1) + separator + str(j + 1) +\
                                       post_bracket + " = " +\
                                       self.get_code(matrix[i, j]) + "\n"

        write_string = write_string + "\n\n"
        # self.file.write(write_string)
        self.write_lines(write_string, prefix="\t")


    def write_function_definition(self, name, args):
        arg_names = ", ".join(str(i) for i in args)
        string = f"def {name}({arg_names}):\n\n"
        self.file.write(string)


    def write_function_returns(self, returns):
        return_names = ", ".join(str(i) for i in returns)
        self.file.write(f"\treturn ({', '.join(return_names)})")


    def close(self):
        self.file.close()