from sympy import Matrix, MatrixSymbol, Expr
from sympy.functions.elementary.trigonometric import InverseTrigonometricFunction
from sympy import acos, sqrt


def vec_norm(vec: Matrix|MatrixSymbol) -> Expr:
    if isinstance(vec, MatrixSymbol):
        vec = vec.as_mutable()
    return sqrt(vec.dot(vec))


def vec_ang_sym(vec1: Matrix|MatrixSymbol, vec2: Matrix|MatrixSymbol) -> InverseTrigonometricFunction:
    """This method finds the angle between two vectors and returns
    the angle in units of radians."""
    if isinstance(vec1, MatrixSymbol):
        vec1 = vec1.as_mutable()
    if isinstance(vec2, MatrixSymbol):
        vec2 = vec2.as_mutable()
    return acos(vec1.dot(vec2) / (vec_norm(vec1) * vec_norm(vec2))) #type:ignore
