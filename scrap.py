import numpy as np
import sympy as sp
import scipy.linalg
from util import unitize
from util import skew
from numpy.linalg import norm
import mpmath
from util import create_C_rot
from util import create_rot_mat
from util import rodriguez_rot



vm = np.array([0, 1, 1])
vmh = unitize(vm)
xx = np.array([1, 0, 0])
yy = np.array([0, 1, 0])
zz = np.array([0, 0, 1])

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
# print(C @ np.array([0, 3, 0]))

C = create_rot_mat(yy, vmh)
# print(C)
# C = rodriguez_rot(-xx, np.radians(45))
# print(np.cos(np.radians(45)))
# print(C)
print("----")
print(C @ np.array([0, 1, 0]))
     
