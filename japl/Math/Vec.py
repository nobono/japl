import numpy as np
# from numpy.linalg import norm



def vec_norm(vec) -> np.ndarray:
    _norm = np.linalg.norm(vec)
    # if _norm == 1.0:
    #     return vec
    # else:
    return (vec / _norm)


def vec_ang(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """This method finds the angle between two vectors and returns
    the angle in units of radians."""
    vec1_norm = np.linalg.norm(vec1)
    vec2_norm = np.linalg.norm(vec2)
    return np.arccos(
            np.dot(vec1, vec2) / (vec1_norm * vec2_norm)
            )
