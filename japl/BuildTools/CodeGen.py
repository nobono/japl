# from sympy.codegen.ast import Assignment
# from sympy.codegen.pyutils import render_as_module
from sympy.printing.pycode import pycode
from japl.BuildTools.DirectUpdate import DirectUpdateSymbol



class CodeGen:


    def __init__(self, expr) -> None:
        print(pycode(expr))


def deball(dynamics, state_vars, input_vars):
    print("\nDYNAMICS\n---------")
    for i, elem in enumerate(dynamics):
        print(i, elem, "\t", pycode(elem, strict=False))

    print("\nSTATE\n---------")
    for i, elem in enumerate(state_vars):
        if isinstance(elem, DirectUpdateSymbol):
            print(i, elem, "\t", pycode(elem.sub_expr, strict=False))
        else:
            print(i, elem, "\t", pycode(elem, strict=False))

    print("\nINPUT\n---------")
    for i, elem in enumerate(input_vars):
        if isinstance(elem, DirectUpdateSymbol):
            print(i, elem, "\t", pycode(elem.sub_expr, strict=False))
        else:
            print(i, elem, "\t", pycode(elem, strict=False))


def deb(array, name: str = ("=" * 50)):
    print(f"\n{name}\n---------")
    for i, elem in enumerate(array):
        if isinstance(elem, DirectUpdateSymbol):
            code = pycode(elem.sub_expr, strict=False).split("\n")  # type:ignore
            code = [i for i in code if '#' not in i]
            print()
            print(i, elem, "\t", code)
        else:
            code = pycode(elem, strict=False).split("\n")  # type:ignore
            code = [i for i in code if '#' not in i]
            print()
            print(i, elem, "\t", code)
