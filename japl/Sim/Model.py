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


    def from_statespace(self,
                        A: np.ndarray,
                        B: np.ndarray,
                        C: Optional[np.ndarray],
                        D: Optional[np.ndarray]) -> "Model":

        self._type = ModelType.StateSpace

        self.A = csr_matrix(A)
        self.B = B
        if C is not None:
            self.C = csr_matrix(C)
        if D is not None:
            self.D = D

        assert self._pre_sim_checks()

        return self


    def _pre_sim_checks(self) -> bool:
        if self._type == ModelType.StateSpace:
            assert self.A.shape == self.C.shape
            assert self.B.shape == self.D.shape
        return True


    def step(self, X: np.ndarray, U: np.ndarray) -> np.ndarray:
        return self.A @ X + self.B @ U
