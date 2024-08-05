# from sympy.core.function import FunctionClass, UndefinedFunction
from typing import Optional
from sympy import Symbol
from sympy import Expr, Matrix, symbols, Function
from sympy.matrices.expressions.matexpr import MatrixElement
from sympy import zeros as sympy_zeros


# class DirectUpdate(tuple):

#     """This class allows for direct settings of the Model state."""

    # def __new__(cls, *args):
    #     args = DirectUpdate.process_args(*args)
    #     return super(DirectUpdate, cls).__new__(cls, args)

        
# class DirectUpdate(Expr):
#     def __new__(cls, var: Symbol|Matrix|Function, val, **assumptions):
#         obj = super().__new__(cls, var, val, **assumptions) #type:ignore
#         obj.subs = DirectUpdate.create_subs(var, val) #type:ignore
#         return obj


class DirectUpdateSymbol(Symbol):


    def __init__(self, name, **assumptions):
        self.state_expr: Optional[Expr] = None
        self.sub_expr: Optional[Expr] = None
        return super().__init__()


class DirectUpdate(Matrix):

    """Takes pair of symbolic definitions which can be added to the symbolic state
    Matrix.
        - for a provided Matrix, each element is converted to a DirectUpdateSymbol
            which contains the "expr" attribute used to update the state during the
            Sim.step()."""

    def __new__(cls, var: Symbol|Matrix|MatrixElement|Function|list, val, **assumptions):
        var = cls.get_symbol(var)
        if var.__class__ in [Symbol, Function]:
            obj = super().__new__(cls, Matrix([var]), **assumptions) #type:ignore
        else:
            obj = super().__new__(cls, var, **assumptions) #type:ignore
        obj.update_map = cls.create_update_map(var, val) #type:ignore
        return obj


    def diff(self, *args, **kwargs) -> Matrix:
        """Direct update of the state will force any derivative
        to zero."""
        return sympy_zeros(*self.shape)


    @staticmethod
    def get_symbol(var):
        """Takes Matrix, Function, Symbol and returns DirectUpdateSymbol."""
        if isinstance(var, Symbol):
            name = f"DU({str(var)})"
            ret = DirectUpdateSymbol(name)
            ret.state_expr = var
            return ret
        match var.__class__():
            case Function():
                name = f"DU({str(var)})"
                ret = DirectUpdateSymbol(name) #type:ignore
                ret.state_expr = var
                return ret
            case Matrix():
                return [DirectUpdate.get_symbol(elem) for elem in var]
            case list():
                return [DirectUpdate.get_symbol(elem) for elem in var]
            case _:
                raise Exception("unhandled case.")


    @staticmethod
    def create_update_map(var, val) -> tuple:
        """sets DirectUpdateSymbol.expr and return update_map"""
        if isinstance(var, DirectUpdateSymbol):
            var.sub_expr = val
            return (var, val)
        match var.__class__():
            case Expr():
                var.sub_expr = val
                return (var, val)
            case Function():
                var.sub_expr = val
                return (var, val)
            case Matrix():
                assert hasattr(val, "__len__")
                for i, j in zip(var, val):
                    i.sub_expr = j
                return [(i, j) for i, j in zip(var, val)] #type:ignore
            case list():
                assert hasattr(val, "__len__")
                for i, j in zip(var, val):
                    i.sub_expr = j
                return [(i, j) for i, j in zip(var, val)] #type:ignore
            case _:
                raise Exception("unhandled case.")


# t = symbols('t')
# x, y, z = symbols('x y z', cls=Function) #type:ignore
# vx, vy, vz = symbols('vx vy vz', cls=Function) #type:ignore
# pos = Matrix([x(t), y(t), z(t)])
# vel = Matrix([vx(t), vy(t), vz(t)])
# # ret = DirectUpdate(pos[0], vel[0])
# # ret = DirectUpdate(x(t), y(t))
# ret = DirectUpdate(pos, vel)
# print(ret)
# print(ret.subs)
# # print(ret.diff(t))
# # print(ret.expr)
# # print(ret, ret.__class__)
# # print(ret.subs)

# x = DirectUpdateSymbol('x')
# y = DirectUpdateSymbol('y')
# z = DirectUpdateSymbol('z')
# p,q,r = symbols("p q r")
# mat = DirectUpdate([x, y, z], [p, q, r])
# pass
