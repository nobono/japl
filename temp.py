import numpy as np
import control as ct
import sympy as sp
from sympy import var
from sympy import Matrix as Mat
from scipy import constants



m, t = sp.symbols('m, t')
omega, alpha = sp.symbols('omega, alpha')
r = sp.MatrixSymbol('r', 3, 1)
T = sp.MatrixSymbol('T', 3, 1)
F = sp.MatrixSymbol('F', 3, 1)
I = sp.MatrixSymbol('I', 3, 3)
g = sp.MatrixSymbol('g', 3, 1)


# alpha = I^-1 (r x F)
# alpha = Mat(I.inv()) * Mat(r).cross(F)


# print(
#     sp.Matrix(r).cross(sp.Matrix(F))
#         )


