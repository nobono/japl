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
alpha = np.array([0., 1])
mach = np.array([0., 5, 10])
axes = {"alpha": alpha,
        "mach": mach}
table = DataTable(data, axes)

# aero = aerotable.AeroTable()
