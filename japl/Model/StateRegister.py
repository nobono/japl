import numpy as np

from sympy import symbols
from sympy import Symbol

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

        self._syms = [v["sym"] for _, v in self.items()]
        self._syms = sorted(self._syms, key=lambda x: self[x.name]["id"])


    def add_state(self, name: str, id: int, label: str = "") -> Symbol:
        """This method registers a SimObject state name and plotting label with a
        user-specified name. The purpose of this register is for ease of access to SimObject
        states without having to use the satte index number.

        -------------------------------------------------------------------
        -- Arguments
        -------------------------------------------------------------------
        -- name - user-specified name of state
        -- id - state index number
        -- label - (optional) string other than "name" that will be displayed
                    in plots / visualization
        -------------------------------------------------------------------
        """
        sym = symbols(name)
        self.update({name: {"id": id, "label": label, "sym": sym}})
        return sym


    def get_sym(self, var: str) -> Symbol:
        return self[var]["sym"]


    def __setitem__(self, key, value) -> None:
        super().__setitem__(key, value)


    def update(self, *args, **kwargs) -> None:
        super().update(*args, **kwargs)
