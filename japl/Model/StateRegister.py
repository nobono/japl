from typing import Optional
import numpy as np

from sympy import Matrix, symbols, MatrixSymbol
from sympy import Symbol, Function
from sympy.matrices.expressions.matexpr import MatrixElement

from japl.BuildTools.DirectUpdate import DirectUpdateSymbol

# ---------------------------------------------------



class StateRegister(dict):

    """
        This class handles the state of a particular model. The purpose of this class
    is to allow for state names, state ids and a symbolic representation of the state
    to be defined for user convenience.

    Use of StateRegister allows for non-linear state-space matrices such as:
        A = [
                [1, 0],
                [0, 'xpos'],
            ]
    """

    def __init__(self, state: dict|list[str]|str = {}):
        self._syms: list[Symbol] = []
        self.matrix_info = {}  # info as full matrices in register

        # if isinstance(state, dict):
        #     self.update(state)
        # elif isinstance(state, list):
        #     self._syms = list(symbols(state))
        # elif isinstance(state, str):
        #     self._syms = list(symbols([state])[0])


    def _pre_sim_checks(self) -> None:
        """This method organizes and processes whatever states were regisered
        by the user. This method must be executed before any simulation runs."""

        # sort the sym variables ordered according to their state index
        self._syms = [v["var"] for _, v in self.items()]
        self._syms = sorted(self._syms, key=lambda x: self[str(x)]["id"])  # type:ignore


    @staticmethod
    def _extract_variable(var) -> Symbol|MatrixSymbol|MatrixElement:
        """This method helps process sympy symbolic variables before
        storing into the register.

        Handle cases when:
            - variable passed is a sympy.Function e.g. "x(t)"
        """
        if isinstance(var, Symbol):
            return var
        elif isinstance(var, Function):
            return Symbol(var.name)  # type:ignore
        elif isinstance(var, MatrixSymbol):
            return var
        elif isinstance(var, MatrixElement):
            return var
        else:
            raise Exception("unhandled case.")


    @staticmethod
    def _extract_variable_name(var) -> str:
        if isinstance(var, MatrixElement):
            name = str(var)
        else:
            name = getattr(var, "name", None)
        if name is None:
            raise Exception("unhandled case: cannot get name attribute from var.")
        return name


    def set(self, vars: tuple|list|Matrix, labels: Optional[list|tuple] = None) -> None:
        """
        register an iterable of sympy Symbols

        Arguments
        ----------
        input_vars:
            iterable of symbolic input variables

        labels:
            (optional) iterable of labels that may be used by the
                    Plotter class. order labels must correspond to order
                    of input_vars.

        Returns
        --------
        (Symbol) - the symbolic object of the state variable
        """
        for id, var in enumerate(vars):  # type:ignore
            # recursive
            if isinstance(var, DirectUpdateSymbol):
                self.set([var.state_expr])

            var = self._extract_variable(var)
            var_name = self._extract_variable_name(var)

            if labels and id < len(labels):
                label = labels[id]
            else:
                label = var_name

            self.update({var_name: {"id": id, "label": label, "var": var, "size": 1}})

            # handle MatrixElement and MatrixSymbol
            self._handle_set_matrix(id, var, var_name, label)


    def _handle_set_matrix(self, id, var, var_name: str, label: str) -> None:
        # NOTE: this process assumes matrices are loaded into the state
        # as a contiguous array.
        if isinstance(var, MatrixElement):
            parent_var = var.parent
            var_name = str(var)
            alias_name = var_name.split('[')[0]  # get mat name before indexing # ]
            if alias_name in self:
                self[alias_name]["size"] += 1
                # NOTE: somehow this increment is being applied
                # to self.matrix_info as well...
            else:
                info = {alias_name: {"id": id, "label": alias_name, "var": parent_var, "size": 1}}
                self.update(info)
                self.matrix_info.update(info.copy())
        elif isinstance(var, MatrixSymbol):
            size = len(var.as_mutable())
            info = {var_name: {"id": id, "label": label, "var": var, "size": size}}
            self.update(info)
            self.matrix_info.update(info.copy())


    def get_ids(self, names: str|list[str]) -> int|list[int]:
        """This method get the sympy variable associated with the provided
        name. variables must first be added to the StateRegister. If a list
        of state names are provided, then a list of corresponding input ids
        will be returned.

        If sympy MatrixElement has been registered in the input e.g. 'x[i, j]',
        then the provided name 'x' will return all indices of that particular
        input.

        -------------------------------------------------------------------
        -- Arguments
        -------------------------------------------------------------------
        -- name - (str | list[str]) name of the symbolic input variable
                name or a list of symbolic input variable names
        -------------------------------------------------------------------
        -- Returns
        -------------------------------------------------------------------
        -- (int | list[int]) - the index of the input variable in the
                input array or list of indices.
        -------------------------------------------------------------------
        """
        if isinstance(names, list):
            # TODO list of names here doesnt account
            # for MatrixSymbols
            ids = [self[k]["id"] for k in names]
            return ids
        else:
            start_id = self[names]["id"]
            size = self[names]["size"]
            var = self[names]["var"]
            if size > 1:
                ids = np.arange(start_id, start_id + size)
                # try to reshape ids to match the MatrixSymbol
                # to qualify for reshaping, the number of ids
                # must equal the expected size of the reshaping.
                # Otherwise, a partial matrix may have been passed
                # which cannot be reshaped to its original shape.
                total_elements = np.prod(var.shape)
                if total_elements == size:
                    shape = var.shape
                    ids = ids.reshape(shape).tolist()
                return ids
            else:
                return start_id


    def get_vars(self) -> Matrix:
        ignores = [MatrixSymbol]
        vars = [i["var"] for i in self.values() if i["var"].__class__ not in ignores]
        return Matrix(vars)


    def get_sym(self, name: str) -> Symbol:
        """This method gets the symbolic variable associated
        with the provided name.

        -------------------------------------------------------------------
        -- Arguments
        -------------------------------------------------------------------
        -- name - (str) name of the symbolic state variable
        -------------------------------------------------------------------
        -- Returns
        -------------------------------------------------------------------
        -- (Symbol) - the symbolic object of the state variable
        -------------------------------------------------------------------
        """
        return self[name]["sym"]


    def __setitem__(self, key, value) -> None:
        super().__setitem__(key, value)


    def update(self, *args, **kwargs) -> None:
        super().update(*args, **kwargs)
