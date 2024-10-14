import sympy as sp
import numpy as np
from japl.BuildTools.CCodeGenerator import CCodeGenerator



dt = sp.symbols("dt")
x, y = sp.symbols("x, y")
a1, a2, a3 = sp.symbols("a1, a2, a3")
b1, b2, b3 = sp.symbols("b1, b2, b3")
A = sp.Matrix([a1, a2, a3])
B = sp.Matrix([b1, b2, b3])
C = sp.MatrixSymbol("C", 3, 3)


y = A * x + C * B
params = [x, a1, a2, a3, B, C]

gen = CCodeGenerator()
gen.add_function(expr=y,
                 params=params,
                 function_name="func",
                 return_name="y")
gen.create_module("simple_model", ".")
