import numpy as np



def vec_ang(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """This method finds the angle between two vectors and returns
    the angle in units of radians."""

    return np.arccos(np.dot(vec1, vec2))
