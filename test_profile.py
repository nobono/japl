import time
import numpy as np
from scipy.interpolate import interpn
from japl import AeroTable



at = AeroTable("./aeromodel/aeromodel.pickle")

method = "linear"

alpha = .1
phi = 0
mach = 0.4
iota = .4

N = 1000
st = time.time()

for i in range(N):
    # n = interpn((at.increments.alpha, at.increments.phi, at.increments.mach), #type:ignore
    #         at._CA_Basic,
    #         [alpha, phi, mach],
    #         method=method)[0]
    at.get_CLMB_Total(alpha, phi, mach, iota)
print((time.time() - st) / N)
