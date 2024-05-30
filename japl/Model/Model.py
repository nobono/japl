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
    def from_statespace(A: np.ndarray,
                        B: np.ndarray,
                        C: Optional[np.ndarray],
                        D: Optional[np.ndarray]) -> "Model":

        model = Model()
        model._type = ModelType.StateSpace

        model.A = csr_matrix(A)
        model.B = B
        if C is not None:
            model.C = csr_matrix(C)
        if D is not None:
            model.D = D
        model.state_dim = model.A.shape[0]

        assert model._pre_sim_checks()

        return model


    def _pre_sim_checks(self) -> bool:
        if self._type == ModelType.StateSpace:
            assert self.A.shape == self.C.shape
            assert self.B.shape == self.D.shape
        return True


    def step(self, X: np.ndarray, U: np.ndarray) -> np.ndarray:
        return self.A @ X + self.B @ U
