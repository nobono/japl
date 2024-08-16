import numpy as np
# from numpy.linalg import norm



def vec_norm(vec) -> np.ndarray:
    _norm = np.linalg.norm(vec)
    # if _norm == 1.0:
    #     return vec
    # else:
    return (vec / _norm)


def vec_ang(vec1: np.ndarray, vec2: np.ndarray, dtype: type = float) -> float:
    """This method finds the angle between two vectors and returns
    the angle in units of radians."""
    dot_product = np.dot(vec1, vec2)
    cross_product_norm = np.linalg.norm(np.cross(vec1, vec2))
    return np.arctan2(cross_product_norm, dot_product, dtype=dtype)
