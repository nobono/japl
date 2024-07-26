import numpy as np



def skew(vec: np.ndarray|list) -> np.ndarray:
    """This method return the skew-symmetric matrix for a vector in R^2 or R^3"""

    _len = len(vec)
    assert _len > 1 and _len <= 3
    if _len == 2:
        return np.array([
            [0, -vec[1]],
            [vec[1], 0]
            ])
    else:
        return np.array([
            [0, -vec[2], vec[1]],
            [vec[2], 0, -vec[0]],
            [-vec[1], vec[0], 0],
            ])
