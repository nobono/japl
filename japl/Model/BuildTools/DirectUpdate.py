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
        self.expr: Optional[Expr] = None
        return super().__init__()


class DirectUpdate(Matrix):


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
        if isinstance(var, Symbol):
            return DirectUpdateSymbol(var.name)
        match var.__class__():
            case Function():
                return DirectUpdateSymbol(var.name) #type:ignore
            case Matrix():
                return [DirectUpdate.get_symbol(elem) for elem in var]
            case list():
                return [DirectUpdate.get_symbol(elem) for elem in var]
            case _:
                raise Exception("unhandled case.")


    @staticmethod
    def create_update_map(var, val) -> tuple:
        if isinstance(var, DirectUpdateSymbol):
            var.expr = val
            return (var, val)
        match var.__class__():
            case Expr():
                var.expr = val
                return (var, val)
            case Function():
                var.expr = val
                return (var, val)
            case Matrix():
                assert hasattr(val, "__len__")
                for i, j in zip(var, val):
                    i.expr = j
                return [(i, j) for i, j in zip(var, val)] #type:ignore
            case list():
                assert hasattr(val, "__len__")
                for i, j in zip(var, val):
                    i.expr = j
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
