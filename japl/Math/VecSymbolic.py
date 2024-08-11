from sympy import Matrix, MatrixSymbol, Expr
from sympy.functions.elementary.trigonometric import InverseTrigonometricFunction
from sympy import acos


def vec_norm(vec: Matrix|MatrixSymbol) -> Expr:
    if isinstance(vec, MatrixSymbol):
        vec = vec.as_mutable()
    return vec.dot(vec)**0.5


def vec_ang_sym(vec1: Matrix|MatrixSymbol, vec2: Matrix|MatrixSymbol) -> InverseTrigonometricFunction:
    """This method finds the angle between two vectors and returns
    the angle in units of radians."""
    if isinstance(vec1, MatrixSymbol):
        vec1 = vec1.as_mutable()
    if isinstance(vec2, MatrixSymbol):
        vec2 = vec2.as_mutable()
    return acos(vec1.dot(vec2) / (vec1.norm() * vec2.norm())) #type:ignore
