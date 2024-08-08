# from sympy.core.function import FunctionClass, UndefinedFunction
from typing import Union
from sympy import Symbol
from sympy import Expr, Matrix, Function, Number
from sympy.matrices.expressions.matexpr import MatrixElement
from sympy import zeros as sympy_zeros

DUType = Union[Symbol, Matrix, MatrixElement, Function, list]



class DirectUpdateSymbol(Symbol):

    """This class inherits from sympy.Symbol so it can be added to the state/input
    matrices. This class allows for direct substitution of a particular state variable
    where state_expr references the state variable and sub_expr references the expression
    or variable which will update the state."""

    def __init__(self, name: str, state_expr: Expr, sub_expr: Expr, **assumptions):
        self.state_expr = state_expr
        self.sub_expr = sub_expr
        return super().__init__()


class DirectUpdate(Matrix):

    """Takes pair of symbolic definitions which can be added to the symbolic state
    Matrix.
        - for a provided Matrix, each element is converted to a DirectUpdateSymbol
            which contains the "expr" attribute used to update the state during the
            Sim.step()."""

    def __new__(cls, var: DUType|str, val: DUType|Expr|float|int|Number, **assumptions):
        var = cls.process_symbols(var, val)
        obj = super().__new__(cls, var, **assumptions) #type:ignore
        return obj


    def diff(self, *args, **kwargs) -> Matrix:
        """Overload diff() of DirectUpdate to force any derivative
        to zero. We do not want the dynamics to update anything directly
        updating the state."""
        return sympy_zeros(*self.shape)


    @staticmethod
    def process_symbols(var, val) -> Matrix:
        """Takes Matrix, Function, Symbol and returns DirectUpdateSymbol."""
        if isinstance(var, Symbol):
            name = str(var)
            return Matrix([DirectUpdateSymbol(name, state_expr=var, sub_expr=val)])
        match var.__class__():
            case str():
                return Matrix([DirectUpdateSymbol(var, state_expr=Symbol(var), sub_expr=val)])
            case Function():
                name = str(var.name)
                return Matrix([DirectUpdateSymbol(name, state_expr=var, sub_expr=val)])
            case Matrix():
                return Matrix([DirectUpdate.process_symbols(var_elem, val_elem)\
                        for var_elem, val_elem in zip(var, val)])
            case list():
                return Matrix([DirectUpdate.process_symbols(var_elem, val_elem)\
                        for var_elem, val_elem in zip(var, val)])
            case _:
                raise Exception("unhandled case.")

