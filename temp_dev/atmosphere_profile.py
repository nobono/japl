import time
import numpy as np
# from scipy.interpolate import interpn
from japl import AeroTable
from japl.Aero.Atmosphere import Atmosphere



atm = Atmosphere()
at = AeroTable("./aeromodel/aeromodel_psb.ma")

method = "linear"

alpha = .1
phi = 0
mach = 0.4
iota = .4

alpha_bound = [at.increments.alpha.min(), at.increments.alpha.max()]
phi_bound = [at.increments.phi.min(), at.increments.phi.max()]
mach_bound = [at.increments.mach.min(), at.increments.mach.max()]
iota_bound = [at.increments.iota.min(), at.increments.iota.max()]

M = 1000
st = time.time()

for i in range(M):
    alpha = np.random.uniform(*alpha_bound)
    phi = np.random.uniform(*phi_bound)
    mach = np.random.uniform(*mach_bound)
    iota = np.random.uniform(*iota_bound)

    sos = atm.speed_of_sound(0)
    # n = interpn((at.increments.alpha, at.increments.phi, at.increments.mach), #type:ignore
    #         at._CA_Basic,
    #         [alpha, phi, mach],
    #         method=method)[0]
    CLMB = at.get_CLMB(alpha, phi, mach, iota)

t = (time.time() - st) / M
print("dt: %.5f, Hz: %.1f" % (t, 1 / t))
