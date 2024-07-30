import numpy as np
from sympy import symbols, Matrix, MatrixSymbol, Expr
from sympy import acos



def vec_ang_sym(vec1: Matrix|MatrixSymbol, vec2: Matrix|MatrixSymbol) -> Expr:
    """This method finds the angle between two vectors and returns
    the angle in units of radians."""
    if isinstance(vec1, MatrixSymbol):
        vec1 = vec1.as_mutable()
    if isinstance(vec2, MatrixSymbol):
        vec2 = vec2.as_mutable()
    return acos(vec1.dot(vec2) / (vec1.norm() * vec2.norm())) #type:ignore
