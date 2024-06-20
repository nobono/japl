# ---------------------------------------------------

from typing import Optional

# ---------------------------------------------------

import numpy as np

# ---------------------------------------------------

# from control.iosys import StateSpace

# ---------------------------------------------------

from enum import Enum

# ---------------------------------------------------

from scipy.sparse import csr_matrix
from scipy.sparse._csr import csr_matrix as Tcsr_matrix
from sympy import Mul, Pow, Symbol, Expr

from japl.Model.StateRegister import StateRegister

# ---------------------------------------------------



class ModelType(Enum):
    NotSet = 0
    StateSpace = 1


class Model:

    """This class is a Model interface for SimObjects"""

    def __init__(self, **kwargs) -> None:
        self._type = ModelType.NotSet
        self.register = StateRegister()
        self.state_dim = 0
        self.A = np.array([])
        self.B = np.array([])
        self.C = np.array([])
        self.D = np.array([])

        # proxy state array updated at each step
        # ***************************************
        # this is ony used during animation or
        # when ODE solver is used step-by-step
        # ***************************************
        self._iX_reference = np.array([])


    def __set_current_state(self, X: np.ndarray):
        """Setter for Model state reference array."""
        self._iX_reference = X


    def get_current_state(self) -> np.ndarray:
        """Getter for Model state reference array. Used to access the
        state array between time steps outside of the Sim class."""
        return self._iX_reference.copy()


    def ss(self,
           A: np.ndarray,
           B: np.ndarray,
           C: Optional[np.ndarray] = None,
           D: Optional[np.ndarray] = None) -> None:

        self._type = ModelType.StateSpace

        self.A = A
        self.B = B

        if C is not None:
            self.C = C
        else:
            self.C = np.eye(self.A.shape[0])

        if D is not None:
            self.D = D
        else:
            self.D = np.zeros_like(self.B)

        self.state_dim = self.A.shape[0]

        # proxy state array which updates statespace matrices
        self._sym_references: list[dict] = []      # info on what matrix slice hold what state ref.


    def _pre_sim_checks(self) -> bool:
        if len(self.A.shape) < 2:
            raise AssertionError("Matrix \"A\" must have shape of len 2")
        if len(self.B.shape) < 2:
            raise AssertionError("Matrix \"B\" must have shape of len 2")
        if self._type == ModelType.StateSpace:
            assert self.A.shape == self.C.shape
            assert self.B.shape == self.D.shape
            # assert len(self._X) == len(self.A)

        # run state-register checks
        self.register._pre_sim_checks()

        # handle user-defined state references in state-space matrices
        self.__handle_sym_expr_references()

        # reduce complexity of mat mult.
        self.A = csr_matrix(self.A)
        self.B = csr_matrix(self.B)
        self.C = csr_matrix(self.C)
        self.D = csr_matrix(self.D)

        # failing here, means matrix contains symbolic references still
        assert self.A.dtype in [float, int]
        assert self.B.dtype in [float, int]
        assert self.C.dtype in [float, int]
        assert self.D.dtype in [float, int]

        return True


    def __handle_sym_expr_references(self) -> None:
        """
            This method handles the user-defined state references (strings) provided
        within the state space model matrices.

        Use of StateRegister allows for non-linear state-space matrices such as:
            A = [
                    [1, 0],
                    [0, 'xpos'],
                ]

        where the registered state corresponds to the following names:
            X = ['xpos', 'ypos', 'ypos']

        The general intended operations ecapsulated by this class are as follows:
            - check user-defined state-space matrix for any strings (state references).

            - store info of [i][j] indices of matrix, the state-references id of the sympy expression
                used, coefficents in expression and powers of sympy state var reference.

        The info gathered in the previous section will be applied in Model.step().
        """

        # TODO this only checks matrix A... expand to B (C? and D?)
        _len = len(self.A)
        for i in range(_len):
            for j in range(_len):
                val = self.A[i, j]
                try:
                    self.A[i][j] = float(val) #type:ignore
                except Exception as _:

                    # split symbolic matrix bin as (coefficent, state-variable)
                    if isinstance(val, Expr):
                        coef_mul: tuple = val.as_coeff_Mul()
                        coef = coef_mul[0]
                        var: Symbol = coef_mul[1] #type:ignore

                        # check if exponents on symbolic state variable exist
                        if isinstance(var, Pow):
                            exp = var.exp
                        else:
                            exp = None

                        # check if state reference is registered
                        if var.name not in self.register:
                            raise Exception(f"User defined state references \"{var.name}\"\
                                    in state-space matrix but \"{var.name}\" is not a registered state")
                        else:
                            # store state references in state-space matrix
                            state_id = self.register[var.name]["id"]
                            self._sym_references.append({"A_slice": (i, j), "state_id": state_id, "coef": coef, "exp": exp})
                            self.A[i:i+1, j:j+1] = float(coef)

        # ensure Model matrices have type float
        # since sympy expressions are allowed in 
        # original definition.
        self.A = self.A.astype(float)


    # TODO this can be generalized to mats B, C, D...
    def __update_A_matrix_exprs(self, A: np.ndarray|Tcsr_matrix, X: np.ndarray) -> None:
        """This method uses the info gathered in "__handle_sym_expr_references()"
        as well as the current state 'X' for the time step and applies the changes
        in matrix A.
        """
        for sim_ref in self._sym_references:
            A_slice = sim_ref["A_slice"]
            state_id = sim_ref["state_id"]
            if sim_ref["exp"] is not None:
                val = X[state_id]**sim_ref["exp"]
            else:
                val = X[state_id]
            A[A_slice[0], A_slice[1]] = sim_ref["coef"] * val


    def step(self, X: np.ndarray, U: np.ndarray) -> np.ndarray:
        """This method is the step method of Model over a single time step.

        -------------------------------------------------------------------
        -- Arguments
        -------------------------------------------------------------------
        -- X - state array for nstep of the model
        -- U - input array for nstep of the model
        -------------------------------------------------------------------
        -- Returns
        -------------------------------------------------------------------
        -- Xdot - derivative of the state (A*X + B*U)
        -------------------------------------------------------------------
        """

        self.__set_current_state(X)
        self.__update_A_matrix_exprs(self.A, X)
        return self.A @ X + self.B @ U


    def get_state_id(self, name: str) -> int:
        """This method get the sympy variable associated with the provided
        name. variables must first be added to the StateRegister.

        -------------------------------------------------------------------
        -- Arguments
        -------------------------------------------------------------------
        -- name - (str) name of the symbolic state variable
        -------------------------------------------------------------------
        -- Returns
        -------------------------------------------------------------------
        -- (int) - the index of the state variable in the state array
        -------------------------------------------------------------------
        """
        return self.register[name]["id"]


    def add_state(self, name: str, id: int, label: str = "") -> Symbol:
        """This method registers a SimObject state name and plotting label
        with a user-specified name. The purpose of this register is for ease
        of access to SimObject states without having to use the satte index
        number.

        -------------------------------------------------------------------
        -- Arguments
        -------------------------------------------------------------------
        -- name - user-specified name of state
        -- id - state index number
        -- label - (optional) string other than "name" that will be displayed
                    in plots / visualization
        -------------------------------------------------------------------
        """
        return self.register.add_state(name=name, id=id, label=label)


    def get_sym(self, var: str) -> Symbol:
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
        return self.register.get_sym(var)


    def print(self):

        def _print_matrix(mat, register = {}):
            if isinstance(mat, Tcsr_matrix):
                mat = mat.toarray()

            print('-' * 50)
            print(name)
            print('-' * 50)
            _shape = mat.shape
            for i in range(_shape[0] - 1):
                if register:
                    print(list(register.keys())[i])
                for j in range(_shape[1] - 1):
                    print(mat[i][j], end=" ")
                print()
            print()

        mat, name = (self.A, "A")
        _print_matrix(mat)
        mat, name = (self.B, "B")
        _print_matrix(mat)

            


