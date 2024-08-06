from sympy import Expr, Piecewise
import sympy as sp



def zero_protect_sym(expr: Expr, condition: Expr) -> Expr:
    """This method will return zero if the provided expession
    is zero."""
    TOLERANCE = 1e-20
    return Piecewise((expr, (condition - 0.0) > TOLERANCE), #type:ignore
                     (0.0, True))

