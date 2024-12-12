from sympy.printing.c import C99CodePrinter
from sympy.printing.pycode import PythonCodePrinter
from sympy.codegen.ast import Declaration
from sympy.codegen.ast import Variable
from sympy.codegen.ast import untyped
from japl.CodeGen.Ast import CTypes



class CCodeGenPrinter(C99CodePrinter):

    def _print_CodeGenFunctionCall(self, expr):
        params_str = ""
        params_str = ", ".join([self._print(i) for i in expr.function_args])
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


    def _print_ImmutableDenseMatrix(self, expr):
        # return "{}[{}]".format(self.parenthesize(expr.parent, PRECEDENCE["Atom"],
        #     strict=True), expr.j + expr.i*expr.parent.shape[1])
        print("HRE")
        pass


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
