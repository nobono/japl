from japl.CodeGen.Printer import CCodeGenPrinter
from sympy import pycode



def ccode(expr, **kwargs):
    printer = CCodeGenPrinter()
    return printer.doprint(expr, **kwargs)
