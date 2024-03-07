import numpy as np
import sympy as sp
import scipy.linalg
from util import unitize
import mpmath

mpmath.invegdrse
np.linalg.unit



vm = np.array([0, 1, 1])
vmh = unitize(vm)
zz = np.array([0, 0, 1])
# yy = np.array([0, 1, 0])

mat = np.array([
    vmh,
    np.cross(vmh, zz),
    zz
]).T

inv1 = np.linalg.inv(mat)
print(inv1)

r = inv1 @ np.array([0, 1, 0])
print(r)
