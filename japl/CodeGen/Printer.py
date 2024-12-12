from sympy import Matrix
from sympy.matrices import ImmutableDenseMatrix, MutableDenseMatrix
from sympy.printing.c import C99CodePrinter
from sympy.printing.pycode import PythonCodePrinter
from sympy.codegen.ast import Declaration
from sympy.codegen.ast import Variable
from sympy.codegen.ast import untyped
from japl.CodeGen.Ast import CTypes
from japl.CodeGen.Util import get_dummy_symbol



class CCodeGenPrinter(C99CodePrinter):

    def _print_CodeGenFunctionCall(self, expr):
        params_str = ""
        # params_str = ", ".join([self._print(i, name="thing") for i in expr.function_args])
        params_list = []
        for arg in expr.function_args:
            if (isinstance(arg, Matrix)
                    or isinstance(arg, MutableDenseMatrix)
                    or isinstance(arg, ImmutableDenseMatrix)):
                params_list += [self._print(arg, name=get_dummy_symbol())]
            else:
                params_list += [self._print(arg)]
        params_str = ", ".join(params_list)

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


    def _print_Kwargs(self, expr):
        kwargs_str = ", ".join(["{" + f"\"{key}\", " + self._print(val) + "}"
                                for key, val in expr.kwpairs.items()])
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

    def _print_CodeGenFunctionCall(self, expr):
        params_str = ""
        params_str = ", ".join([self._print(i) for i in expr.function_args])
        if len(expr.function_args) and len(expr.function_kwargs):
            params_str += ", "
        if (kwargs_str := expr._dict_to_kwargs_str(expr.function_kwargs)):
            params_str += f"{kwargs_str}"
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


