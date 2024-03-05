import numpy as np



def pronav(X, r_targ, v_targ, N=4.0):
    rm = X[:3]
    vm = X[3:6]
    v_r = v_targ - vm
    r = r_targ - rm
    omega = np.cross(r, v_r) / np.dot(r, r)
    ac = N * np.cross(v_r, omega)
    return ac
