import numpy as np
from japl.DataTable.DataTable import DataTable
import model
import aerotable



data = np.array([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]],
                dtype=float)
axes = {"alpha": np.array([0., 1, 2]),
        "mach": np.array([0., 1, 2])}

table = DataTable(data, axes)

aero = aerotable.AeroTable(CA=table)

ret = aero.get_CA(alpha=1, mach=1)
print(ret)


# aero.set_CA(table)
# model.set_aerotable
