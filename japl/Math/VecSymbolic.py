from sympy import Matrix, MatrixSymbol, Expr
from sympy.functions.elementary.trigonometric import InverseTrigonometricFunction
from sympy import atan2


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
    dot_product = vec1.dot(vec2)
    cross_product_norm = vec1.cross(vec2).norm()
    return atan2(cross_product_norm, dot_product)  # type:ignore
