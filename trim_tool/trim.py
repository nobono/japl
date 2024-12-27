import numpy as np
from japl import AeroTable
from japl import Atmosphere
from japl import JAPL_HOME_DIR



atmos = Atmosphere()
aero = AeroTable(JAPL_HOME_DIR + "/aerodata/aeromodel_psb.mat")


class AeroTool:


    def __init__(self, aerotable: AeroTable) -> None:
        self.aerotable = aerotable


    def call_aero(self, alpha, mach, alt, iota, boost: bool = False, tof: float = 0):
        sos = atmos.speed_of_sound(alt)
        vel = mach * sos
        q_bar = atmos.dynamic_pressure(vel=vel, alt=alt)

        Sref = aero.get_Sref()
        Lref = aero.get_Lref()

        CA = aero.get_CA(alpha=alpha, mach=mach, iota=iota, thrust=0)
        CN = aero.get_CNB(alpha=alpha, mach=mach, iota=iota)
        CLM = aero.get_CLMB(alpha=alpha, mach=mach, iota=iota)
        aero.get_CA

        f_vec = (q_bar * Sref) * -np.array([CA, CN])
        My = (q_bar * Sref * Lref) * np.array([CLM])
        # TODO cg
        cg = 1.0

        My = My - f_vec[1] * cg
        return (f_vec, My)


    def trim_3dof(self, alpha, mach) -> float:
        pass


    def trim(self):
        alphas = self.aerotable.increments.alpha
        machs = self.aerotable.increments.mach
        alts = self.aerotable.increments.alt
        iotas = self.aerotable.increments.iota

        nalpha = len(alphas)
        nmach = len(machs)
        nalt = len(alts)
        niota = len(iotas)

        trim_iota = np.empty(shape=(nalpha, nmach))
        trim_Az = np.empty(shape=(nalpha, nmach, nalt))
        trim_q = np.empty(shape=(nalpha, nmach, nalt))
        trim_iota.fill(np.nan)
        trim_Az.fill(np.nan)
        trim_q.fill(np.nan)

        for ialpha, alpha in enumerate(alphas):
            for imach, mach in enumerate(machs):
                for iiota, iota in enumerate(iotas):
                    if np.isnan(trim_iota[ialpha, imach]):
                        iota = self.trim_3dof(alpha=alpha, mach=mach)
                    else:
                        iota = trim_iota[ialpha, imach]

                    if not np.isnan(iota):
                        for ialts, alt in enumerate(alts):
                            pass


tool = AeroTool(aero)
tool.trim()
