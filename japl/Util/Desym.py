from typing import Any
import numpy as np
import inspect
from sympy import Expr
from sympy import Symbol
from sympy import Matrix
from sympy import Piecewise
from sympy import Function
from sympy import lambdify



def Max_(a, b):
    """vectorized sympy.Max()"""
    return Piecewise((b, a < b), (a, True))


def Min_(a, b):
    """vectorized sympy.Min()"""
    return Piecewise((a, a < b), (b, True))


class Desym:

    custom_lambdify_dict = {
            'Matrix': np.array,
            'ImmutableMatrix': np.array,
            'MutableDenseMatrix': np.array,
            'ImmutableDenseMatrix': np.array,
            'Max': Max_,
            'Min': Min_,
            }

    def __init__(self,
                 vars: Symbol|tuple[Any, ...]|list,
                 func: Expr|Matrix|Function,
                 dummify: bool = False,
                 cse: bool = True,
                 modules: dict|list[dict] = {},
                 array_arg: bool = False) -> None:
        self.modules = {}
        self.modules.update(self.custom_lambdify_dict)
        if isinstance(modules, dict):
            self.modules.update(modules)
        elif isinstance(modules, list):  # type:ignore
            for module in modules:
                self.modules.update(module)
        else:
            raise Exception("modules must be dict or list of dicts.")
        self.array_arg = array_arg      # option to pass args as single array
        if isinstance(vars, Symbol):
            self.vars = (vars,)
        else:
            self.vars = tuple(vars)
        self.f = lambdify(self.vars, func,
                          modules=[self.modules, "numpy"],
                          dummify=dummify,
                          cse=cse)
        self.code = inspect.getsource(self.f)


    def __call__(self, *args) -> np.ndarray:
        if self.array_arg:
            args = [*args[0]]       # unpack if option set
        return self.f(*args)


    def dump(self, f):
        """This method dumps the lambdify'd sympy expression
        code string to a provided file."""
        try:
            f.write(self.code)
        except Exception as e:
            raise Exception(e)
