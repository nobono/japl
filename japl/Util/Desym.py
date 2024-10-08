from typing import Any, Callable
import numpy as np
import inspect

from japl.BuildTools.DirectUpdate import DirectUpdateSymbol
from japl.Util.Util import flatten_list
from sympy import Expr
from sympy import Symbol
from sympy import Matrix
from sympy import Piecewise
from sympy import Function
from sympy import MatrixSymbol
from sympy.matrices.expressions.matexpr import MatrixElement
from sympy import lambdify
from sympy.utilities.autowrap import autowrap
from copy import deepcopy



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
                 expr: Expr|Matrix|Function,
                 dummify: bool = False,
                 cse: bool = True,
                 modules: dict|list[dict] = {},
                 is_array_arg: bool = False,
                 wrap_type: str = "lambdify") -> None:
        self.modules = self.process_modules(modules)
        self.vars = self.process_vars(vars)
        self.is_array_arg = is_array_arg      # option to pass args as single array
        # create wrapped Callable
        self.f = self.process_function_wrap(expr=expr, vars=self.vars, modules=self.modules,
                                            wrap_type=wrap_type, dummify=dummify, cse=cse)
        # store function code for debugging
        self.code = inspect.getsource(self.f)


    @staticmethod
    def process_function_wrap(expr: Expr|Matrix|Function, vars: tuple, modules: dict, wrap_type: str,
                              dummify: bool, cse: bool):
        """This method handled the wrapping of symbolic expressions to a
        callable object. currently, sympy's lambdify and autowrap are
        supported."""
        match wrap_type:
            case "lambdify":
                # -----------------------------------------------------
                # NOTE:
                # -----------------------------------------------------
                # MatrixSymbols must be passed to lambdify
                # in order to use MatrixElements inside of an
                # expression. The model state may consist of
                # MatrixElements but the parent MatrixSymbol
                # must be passed as an argument.
                #
                # _matrix_register must gather all MatrixElements
                # of the same MatrixSymbol.
                # -----------------------------------------------------
                is_iterable = hasattr(expr, "__len__")

                _matrix_register = []
                del_ids = []

                # get state_vars from argument tuple
                # arguments tuple has signature of:
                #   (t, state_vars, input_vars, static_vars, dt)
                state_vars = deepcopy(vars[1])

                if is_iterable:
                    for i, var in enumerate(state_vars):
                        if isinstance(var, DirectUpdateSymbol):
                            if isinstance(var.state_expr, MatrixElement):
                                parent = var.state_expr.parent
                                del_ids += [i]
                                if parent not in _matrix_register:
                                    _matrix_register += [parent]

                    # safely remove MatrixElement rows
                    # iterively from the bottom up
                    del_ids.sort()
                    for id in reversed(del_ids):
                        state_vars.row_del(id)

                    # pass vars as a list of [Symbol, MatrixSymbols, ...]
                    state_vars = state_vars.flat() + _matrix_register

                    # unpack vars into mutable iter
                    # and replace state_vars (index 1)
                    vars = [*vars]  # type:ignore
                    vars[1] = state_vars  # type:ignore
                # -----------------------------------------------------

                ret = lambdify(vars, expr,
                               modules=[modules, "numpy"],
                               dummify=dummify,
                               cse=cse)
            case "autowrap":
                # TODO this is unfinished
                # unpack vars
                _vars = []
                for var in flatten_list(vars):
                    if isinstance(var, DirectUpdateSymbol):
                        _vars += [var.state_expr]
                    else:
                        _vars += [var]
                # replace aero.get_CNB w/ modules['aerotable_get_CNB']
                ret = autowrap(expr=expr,
                               args=_vars,
                               language="C",
                               include_dirs=[],
                               # library_dirs=
                               # libraries=,
                               verbose=False,
                               backend="cython",
                               # tempdir=outdir,
                               # code_gen=code_gen,
                               )
            case _:  # default to lambdify
                ret = lambdify(vars, expr,
                               modules=[modules, "numpy"],
                               dummify=dummify,
                               cse=cse)
        return ret


    @staticmethod
    def process_vars(vars) -> tuple:
        """This method handles vars being pass and ensures
        self.vars is a tuple of Symbols."""
        if isinstance(vars, Symbol):
            ret = (vars,)
        else:
            ret = tuple(vars)
        return ret


    @staticmethod
    def process_modules(modules: dict|list[dict]) -> dict:
        """This method handles modules passed as either
        a dict or list[dict]."""
        ret = {}
        ret.update(Desym.custom_lambdify_dict)
        if isinstance(modules, dict):
            ret.update(modules)
        elif isinstance(modules, list):  # type:ignore
            for module in modules:
                ret.update(module)
        else:
            raise Exception("modules must be dict or list of dicts.")
        return ret


    def __call__(self, *args) -> np.ndarray:
        if self.is_array_arg:
            args = [*args[0]]       # unpack if option set
        # TODO: for completing Matrix in state capability:
        # args need to be re-formatted (reshaped) for any
        # matrix being passed to the lambdify'd function
        # args = (0, [*np.zeros(16), np.eye(16)], np.zeros(6), np.zeros(6), 0.01)
        return self.f(*args)


    def dump(self, f):
        """This method dumps the lambdify'd sympy expression
        code string to a provided file."""
        try:
            f.write(self.code)
        except Exception as e:
            raise Exception(e)
