import numpy as np
import sympy as sp
import scipy.linalg
from util import unitize
from util import skew
from numpy.linalg import norm
import mpmath
from util import create_C_rot



vm = np.array([4, 1, 2])
vmh = unitize(vm)
zz = np.array([0, 0, 1])
yy = np.array([0, 1, 0])

mat = np.array([
    vmh,
    np.cross(vmh, zz),
    zz
]).T

# inv1 = np.linalg.inv(mat)
# r = inv1 @ np.array([0, 1, 0])
# print(r)
# print(norm(r))

C = create_C_rot(vm)
print(C @ np.array([0, 3, 0]))
     
