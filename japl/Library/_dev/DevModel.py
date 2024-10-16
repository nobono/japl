import sympy as sp
import numpy as np
from japl.BuildTools.CCodeGenerator import CCodeGenerator
from japl import JAPL_HOME_DIR



x, y = sp.symbols("x, y")
a1, a2, a3 = sp.symbols("a1, a2, a3")
b1, b2, b3 = sp.symbols("b1, b2, b3")
A = sp.Matrix([a1, a2, a3])
B = sp.Matrix([b1, b2, b3])
C = sp.MatrixSymbol("C", 3, 3)

expr = A * x + C * B
params = [x, a1, a2, a3, B, C]

truth = expr.subs({
    x: 1.,
    a1: 1.,
    a2: 2.,
    a3: 3.,
    b1: 1.,
    b2: 2.,
    b3: 3.,
    C: sp.Matrix(np.array([[1, 2, 3],
                           [4, 5, 6],
                           [7, 8, 9]], dtype=float))
    }).doit().flat()


if __name__ == "__main__":

    gen = CCodeGenerator()
    gen.add_function(expr=expr,
                     params=params,
                     function_name="func",
                     return_name="y")
    gen.create_module("_dev_model", f"{JAPL_HOME_DIR}/japl/Library/_dev")
