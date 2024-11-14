import numpy as np
from japl.DataTable.DataTable import DataTable
from japl.Interpolation.Interpolation import LinearInterp
# import atmosphere
import aerotable
import linterp


data = np.array([[1., 2, 3],
                 [-1., -2, -3]])
axes = {"alpha": np.array([0., 1]),
        "mach": np.array([0., 5, 10])}

_axes = tuple([*axes.values()])
# lt = linterp.Interp2d(_axes, data)
lt = LinearInterp(_axes, data)

dt = DataTable(data, axes)
# dt(alpha=.2, mach=5)
# print(dt(alpha=.2, mach=5))  # 1.2
# print(dt.axes)

# print(dt.interp.ndim)
# aero = aerotable.AeroTable(CA=dt.interp)
# print(lt.interp.__class__)

aero = aerotable.AeroTable(CA=lt.interp, CNB=lt.interp)
print(aero.CA((1, 1)))
print(aero.CNB((2, 2)))
# print(aero.table_info)
# print(aero.get_CA())
