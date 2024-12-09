from sympy.printing.c import C99CodePrinter
from sympy.codegen.ast import Declaration
from sympy.codegen.ast import Variable
from sympy.codegen.ast import untyped



class CCodeGenPrinter(C99CodePrinter):

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
        # if params != None:  # Must be "!= None", cannot be "is not None" # noqa
        #     result += ' = %s' % self._print(params)
        return result
