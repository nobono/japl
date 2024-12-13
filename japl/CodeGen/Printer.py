from sympy.printing.c import C99CodePrinter
from sympy.printing.pycode import PythonCodePrinter
from sympy.codegen.ast import Declaration
from sympy.codegen.ast import Variable
from sympy.codegen.ast import untyped
from japl.CodeGen.Ast import CTypes
from japl.CodeGen.Ast import convert_symbols_to_variables
from japl.CodeGen.JaplFunction import JaplFunction



class CCodeGenPrinter(C99CodePrinter):

    code_str: str = 'c'

    def _print_Function(self, expr):
        # ---------------------------------------------------------------
        # NOTE: CodeGen printer does not recognize _print_JaplFunction
        # JaplFunction print logic must be handled here.
        # ---------------------------------------------------------------
        if isinstance(expr, JaplFunction):
            return self._print_JaplFunction(expr)
        else:
            return super()._print_Function(expr)


    def _print_JaplFunction(self, expr: JaplFunction):
        # if here, printing the call of JaplFunction
        # ---------------------------------------------------------------
        # NOTE: this is not very clean but _get_parameter_variables must
        # know the codegen language type and thus parameters cannot be
        # defined in JaplFuction or CodeGenFunctionCall constructors.
        # ---------------------------------------------------------------
        return self._print_CodeGenFunctionCall(expr.codegen_function_call)


    def _print_CodeGenFunctionCall(self, expr):
        parameters = convert_symbols_to_variables(expr.function_args, self.code_str)
        if not hasattr(parameters, "__len__"):
            parameters = [parameters]
        params_str = ", ".join([self._print(i.symbol) for i in parameters])  # type:ignore

        if len(expr.function_args) and len(expr.function_kwargs):
            params_str += ", "
        kwargs_list = []
        for key, val in expr.function_kwargs.items():
            kwargs_list += ["{" + f"\"{key}\", {val}" + "}"]
        if kwargs_list:
            params_str += "{" + ", ".join(kwargs_list) + "}"
        return f"{expr.name}({params_str})"


    def _print_Constructor(self, expr):
        params = expr.parameters
        # handle both Declaration and Variables passed
        if isinstance(expr.variable, Declaration):
            var = expr.variable.variable
        else:
            var = expr.variable

        if var.type == untyped:
            raise ValueError("C does not support untyped variables")

        elif isinstance(var, Variable):
            result = '{t} {s}({p})'.format(
                t=self._print(var.type),
                s=self._print(var.symbol),
                p=", ".join([self._print(p) for p in params])
            )
        else:
            raise NotImplementedError("Unknown type of var: %s" % type(var))
        return result


    def _print_ImmutableDenseMatrix(self, expr, name: str = ""):
        # NOTE: name keyword arg must be passed. This is typically
        # passed from the CodeGen Printer method "_print()".
        matrix_type = self._print(CTypes.from_expr(expr).as_vector().as_ref())
        return f"{matrix_type} {name}"


    def _print_MutableDenseMatrix(self, expr, name: str = ""):
        # NOTE: name keyword arg must be passed. This is typically
        # passed from the CodeGen Printer method "_print()".
        matrix_type = self._print(CTypes.from_expr(expr).as_vector().as_ref())
        return f"{matrix_type} {name}"


    def _print_Kwargs(self, expr):
        kwargs_str = ""
        for key, val in expr.kwpairs.items():
            val_variable = convert_symbols_to_variables(val, self.code_str)
            kwargs_str += "{" + f"\"{key}\", " + self._print(val_variable) + "}, "
        kwargs_str = kwargs_str.strip(", ")
        kwargs_str = "{" + kwargs_str + "}"
        return kwargs_str


    def _print_Dict(self, expr):
        dict_str = ", ".join(["{" + f"\"{key}\", " + self._print(val) + "}"
                              for key, val in expr.kwpairs.items()])
        dict_str = dict_str.strip(", ")
        dict_str = "{" + dict_str + "}"
        return dict_str


class PyCodeGenPrinter(PythonCodePrinter):

    code_str: str = "py"

    def _print_Function(self, expr):
        # ---------------------------------------------------------------
        # NOTE: CodeGen printer does not recognize _print_JaplFunction
        # JaplFunction print logic must be handled here.
        #
        # super()._print_Function(expr) is not supported by PythonCodePrinter
        # ---------------------------------------------------------------
        return self._print_JaplFunction(expr)



    def _print_JaplFunction(self, expr: JaplFunction):
        # if here, printing the call of JaplFunction
        # ---------------------------------------------------------------
        # NOTE: this is not very clean but _get_parameter_variables must
        # know the codegen language type and thus parameters cannot be
        # defined in JaplFuction or CodeGenFunctionCall constructors.
        # ---------------------------------------------------------------
        # parameters = expr._get_parameter_variables("py")  # first get params
        return self._print_CodeGenFunctionCall(expr.codegen_function_call)


    def _print_CodeGenFunctionCall(self, expr):
        parameters = convert_symbols_to_variables(expr.function_args, self.code_str)
        if not hasattr(parameters, "__len__"):
            parameters = [parameters]
        params_str = ", ".join([self._print(i.symbol) for i in parameters])  # type:ignore

        if len(expr.function_args) and len(expr.function_kwargs):
            params_str += ", "
        params_str += expr._dict_to_kwargs_str(expr.function_kwargs)
        return f"{expr.name}({params_str})"


    def _print_Kwargs(self, expr):
        kwargs_str = ", ".join([f"{key}={self._print(val)}" for key, val in expr.kwpairs.items()])
        kwargs_str = kwargs_str.strip(", ")
        return kwargs_str


    def _print_Dict(self, expr):
        kwargs_str = ", ".join([f"{key}: {self._print(val)}" for key, val in expr.kwpairs.items()])
        kwargs_str = kwargs_str.strip(", ")
        kwargs_str = "{" + kwargs_str + "}"
        return kwargs_str


def ccode(expr, **kwargs):
    printer = CCodeGenPrinter()
    return printer.doprint(expr, **kwargs)


def pycode(expr, **kwargs):
    printer = PyCodeGenPrinter()
    return printer.doprint(expr, **kwargs)
