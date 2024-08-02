from typing import Optional
import numpy as np

from sympy import Matrix, symbols
from sympy import Symbol, Function

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

        if isinstance(state, dict):
            self.update(state)
        elif isinstance(state, list):
            self._syms = list(symbols(state))
        elif isinstance(state, str):
            self._syms = list(symbols([state])[0])


    def _pre_sim_checks(self) -> None:
        """This method organizes and processes whatever states were regisered
        by the user. This method must be executed before any simulation runs."""

        # sort the sym variables ordered according to their state index
        self._syms = [v["var"] for _, v in self.items()]
        self._syms = sorted(self._syms, key=lambda x: self[str(x)]["id"])


    def __process_variables(self, var) -> Symbol:
        """This method helps process sympy symbolic variables before
        storing into the register.

        Handle cases when:
            - variable passed is a sympy.Function e.g. "x(t)"
        """
        if isinstance(var, Symbol):
            return var
        elif isinstance(var, Function):
            return Symbol(var.name) #type:ignore
        else:
            raise Exception("unhandled case.")


    @DeprecationWarning
    def add_state(self, name: str, id: int, label: str = "") -> Symbol:
        """This method registers a SimObject state name and plotting label with a
        user-specified name. The purpose of this register is for ease of access to SimObject
        states without having to use the state index number.

        -------------------------------------------------------------------
        -- Arguments
        -------------------------------------------------------------------
        -- name - user-specified name of state
        -- id - state index number
        -- label - (optional) string other than "name" that will be displayed
                    in plots / visualization
        -------------------------------------------------------------------
        """
        var = symbols(name)
        self.update({name: {"id": id, "label": label, "var": var}})
        return var


    def set(self, vars: tuple|list|Matrix, labels: Optional[list|tuple] = None):
        """register state and labels"""
        for id, var in enumerate(vars): #type:ignore
            var = self.__process_variables(var)
            var_name = str(var)
            if labels and id < len(labels):
                label = labels[id]
            else:
                label = var_name
            self.update({var_name: {"id": id, "label": label, "var": var}})


    def get_vars(self) -> Matrix:
        return Matrix([i["var"] for i in self.values()])


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
