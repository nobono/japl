import numpy as np
from japl import AeroTable
from scipy.interpolate import RegularGridInterpolator
from time import perf_counter
import linterp


np.random.seed(123)
DIR = "/home/david/work_projects/control/aeromodel/stage_1_aero.mat"
# stage_1 = AeroTable(DIR, from_template="orion")
aerotable = AeroTable(DIR, from_template="orion")

# aerotable.add_stage(stage_1)

# a = np.linspace(0, 10, 20)
# b = np.linspace(-1, 1, 20)
# c = np.linspace(-1, 1, 20)
# axes = (a, b, c)
# table = np.array([[1,2,3],
#                   [4,5,6],
#                   [7,8,9]])
# table = np.random.random((20, 20, 20))

axes = (aerotable.CNB.axes["alpha"],
        aerotable.CNB.axes["mach"],
        aerotable.CNB.axes["alt"])
table = aerotable.CNB

point = np.array([[0.1, 0.2, 0.3],
                  [0.4, 0.5, 0.6]])

N = 1
p = linterp.Interp3d(axes, table)

t1 = np.eye(3)
t2 = np.array([[1, 2, 3],
               [4, 5, 6],
               [7, 8, 9]]) * 0.1
t3 = np.random.random((3, 3))
tup = (t1, t2, t3)
ret = p.interpolate(tup)

quit()

# st = perf_counter()
# for i in range(N):
#     ret = p.interpolate(point)
# exec = (perf_counter() - st)

# st = perf_counter()
rgi = RegularGridInterpolator(axes, table)
for i in range(N):
    # ret2 = aerotable.CNB(alpha=np.array([.1, .4]),
    #                      mach=np.array([.2, .5]),
    #                      alt=np.array([.3, .6]))
    ret2 = rgi(point)
# exec2 = (perf_counter() - st)
print(ret2[0], ret2[1])

# linterp.test_dict({"a": 1, "b": 2, "c": 3})

# print(exec, exec2)
# print(ret, ret2)




