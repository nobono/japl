from typing import Any
import numpy as np
from sympy import Expr
from sympy import Matrix
from sympy import Piecewise
from sympy import lambdify



Max_ = lambda a, b: Piecewise((b, a < b), (a, True))      # vectorized sympy.Max()
Min_ = lambda a, b: Piecewise((a, a < b), (b, True))      # vectorized sympy.Min()


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
                 vars: tuple[Any, ...]|list,
                 func: Expr|Matrix,
                 dummify: bool = False,
                 cse: bool = True,
                 array_arg: bool = False) -> None:
        self.array_arg = array_arg      # option to pass args as single array
        self.vars = tuple(vars)
        self.f = lambdify(self.vars, func,
                          modules=[self.custom_lambdify_dict, "numpy"],
                          dummify=dummify,
                          cse=cse)

    def __call__(self, *args) -> np.ndarray:
        if self.array_arg:
            args = [*args[0]]       # unpack if option set
        return self.f(*args)
