import numpy as np
from japl.DataTable.DataTable import DataTable
from japl.Interpolation.Interpolation import LinearInterp
from scipy.interpolate import RegularGridInterpolator
from japl import AeroTable
# import atmosphere
import aerotable
import linterp

aero_file = f"aeromodel/aeromodel_psb.mat"
aero = AeroTable(aero_file)
alts = np.linspace(0, 30_000, 100)


data = np.array([[1., 2, 3],
                 [-10., -20, -30]])
daxes = {"alpha": np.array([0., 1]),
         "mach": np.array([0., 5, 10])}
axes = tuple([*daxes.values()])

table = DataTable(data, daxes)
# print(d.axes['alpha'])
# table = table.mirror_axis(0)
# print(d.axes['alpha'])
# print(d(alpha=1, mach=0))

# data = np.array([
#     [np.array([0., 1, 2]),
#      np.array([0., 1, 2]) * 10],
#     [np.array([1., 2, 3]),
#      np.array([1., 2, 3]) * 10],
#     ])
# axes = (np.array([0., 1]),
#         np.array([0., 10]),
#         np.array([0., 100, 200]),
#         )
# # dt = DataTable(data, axes)
# # lt = LinearInterp(axes, data, True)
# # rg = LinearInterp(axes, data, False)

# t1 = aero.CNB
# t2 = t1.copy()
# lt = LinearInterp(tuple([*t2.axes.values()]), np.array(t2), True)
# t2.interp = lt  # type:ignore

quit()
diff_arg = "alpha"
delta_alpha = np.radians(0.1)
delta_arg = delta_alpha


max_alpha = max(table.axes.get(diff_arg))  # type:ignore
min_alpha = min(table.axes.get(diff_arg))  # type:ignore

# handle table args
val_args = AeroTable._get_table_args(table, **table.axes)
arg_grid = np.meshgrid(*val_args, indexing="ij")
args = {str(k): v for k, v in zip(table.axes, arg_grid)}

# create diff_arg plus & minus values for linear interpolation
args_plus = args.copy()
args_minus = args.copy()
args_plus[diff_arg] = np.clip(args[diff_arg] + delta_arg, min_alpha, max_alpha)
args_minus[diff_arg] = np.clip(args[diff_arg] - delta_arg, min_alpha, max_alpha)

val_plus = table(**args_plus)
# print(table(alpha=.05, phi=0, mach=1, iota=.01))

val_minus = table(**args_minus)
diff_table = (val_plus - val_minus) / (args_plus[diff_arg] - args_minus[diff_arg])  # type:ignore

dtnew = DataTable(diff_table, axes=table.axes)
print(dtnew[:])


quit()


# aero = aerotable.AeroTable(CA=lt.interp, CNB=lt.interp)
# print(aero.CA((1, 1)))
# print(aero.CNB((2, 2)))
