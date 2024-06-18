# ---------------------------------------------------

from typing import Optional

# ---------------------------------------------------

import numpy as np

# ---------------------------------------------------

from control.iosys import StateSpace

# ---------------------------------------------------

from enum import Enum

# ---------------------------------------------------

from scipy.sparse import csr_matrix
from scipy.sparse._csr import csr_matrix as Tcsr_matrix



class ModelType(Enum):
    NotSet = 0
    StateSpace = 1


class Model:

    """This class is a Model interface for SimObjects"""

    def __init__(self) -> None:
        self._type = ModelType.NotSet
        self.state_dim = 0
        self.A = np.array([])
        self.B = np.array([])
        self.C = np.array([])
        self.D = np.array([])


    @staticmethod
    def ss(A: np.ndarray,
           B: np.ndarray,
           C: Optional[np.ndarray] = None,
           D: Optional[np.ndarray] = None) -> "Model":

        model = Model()
        model._type = ModelType.StateSpace

        model.A = csr_matrix(A)
        model.B = B

        if C is not None:
            model.C = csr_matrix(C)
        else:
            model.C = csr_matrix(np.eye(model.A.shape[0]))

        if D is not None:
            model.D = D
        else:
            model.D = np.zeros_like(model.B)

        model.state_dim = model.A.shape[0]

        assert model._pre_sim_checks()

        return model


    def _pre_sim_checks(self) -> bool:
        if len(self.A.shape) < 2:
            raise AssertionError("Matrix \"A\" must have shape of len 2")
        if len(self.B.shape) < 2:
            raise AssertionError("Matrix \"B\" must have shape of len 2")
        if self._type == ModelType.StateSpace:
            assert self.A.shape == self.C.shape
            assert self.B.shape == self.D.shape
        return True


    def step(self, X: np.ndarray, U: np.ndarray) -> np.ndarray:
        return self.A @ X + self.B @ U


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

            


