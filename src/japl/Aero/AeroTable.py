import numpy as np
from scipy.interpolate import interpn
from scipy.io import loadmat
import pickle
from astropy import units as u



__ft2m = (1.0 * u.imperial.foot).to_value(u.m) #type:ignore
__inch2m = (1.0 * u.imperial.inch).to_value(u.m) #type:ignore
__deg2rad = (np.pi / 180.0)
__lbminch2Nm = (1.0 * u.imperial.lbm * u.imperial.inch**2).to_value(u.kg * u.m**2) #type:ignore


class Increments:
    alpha = np.empty([])
    phi = np.empty([])
    mach = np.empty([])
    alt = np.empty([])
    iota = np.empty([])
    iota_prime = np.empty([])

    nalpha = 0
    nphi = 0
    nmach = 0
    nalt = 0
    niota = 0
    niota_prime = 0

    def __repr__(self) -> str:
        members = [i for i in dir(self) if "__" not in i]
        return "\n".join(members)


class AeroTable:

    """This class is for containing Aerotable data for a particular SimObject."""

    def __init__(self, data: str|dict) -> None:
        data_dict = {}
        if isinstance(data, str):
            self.__path = data
            assert ".pickle" in self.__path
            with open(self.__path, "rb") as f:
                data_dict = pickle.load(f)
            
        _increments = data_dict.get("increments", None)
        if _increments:
            self.increments = Increments()
            self.increments.alpha       = _increments.get("alpha", None)
            self.increments.phi         = _increments.get("phi", None)
            self.increments.mach        = _increments.get("mach", None)
            self.increments.alt         = _increments.get("alt", None)
            self.increments.iota        = _increments.get("iota", None)
            self.increments.iota_prime  = _increments.get("iota_prime", None)
            self.increments.nalpha       = len(self.increments.alpha)
            self.increments.nphi         = len(self.increments.phi)
            self.increments.nmach        = len(self.increments.mach)
            self.increments.nalt         = len(self.increments.alt)
            self.increments.niota        = len(self.increments.iota)
            self.increments.niota_prime  = len(self.increments.iota_prime)

        _psb = data_dict.get("psb", None)
        if _psb:
            self._CA_inv     = _psb.get("CA_inv",     None)   # (alpha, phi, mach)
            self._CA_Basic   = _psb.get("CA_Basic",   None)   # (alpha, phi, mach)
            self._CA_0_Boost = _psb.get("CA_0_Boost", None)   # (phi, mach, alt)
            self._CA_0_Coast = _psb.get("CA_0_Coast", None)   # (phi, mach, alt)
            self._CA_IT      = _psb.get("CA_IT",      None)   # (alpha, phi, mach, iota)
            self._CYB_Basic  = _psb.get("CYB_Basic",  None)   # (alpha, phi, mach)
            self._CYB_IT     = _psb.get("CYB_IT",     None)   # (alpha, phi, mach, iota)
            self._CNB_Basic  = _psb.get("CNB_Basic",  None)   # (alpha, phi, mach)
            self._CNB_IT     = _psb.get("CNB_IT",     None)   # (alpha, phi, mach, iota)
            self._CLLB_Basic = _psb.get("CLLB_Basic", None)   # (alpha, phi, mach)
            self._CLLB_IT    = _psb.get("CLLB_IT",    None)   # (alpha, phi, mach, iota)
            self._CLMB_Basic = _psb.get("CLMB_Basic", None)   # (alpha, phi, mach)
            self._CLMB_IT    = _psb.get("CLMB_IT",    None)   # (alpha, phi, mach, iota)
            self._CLNB_Basic = _psb.get("CLNB_Basic", None)   # (alpha, phi, mach)
            self._CLNB_IT    = _psb.get("CLNB_IT",    None)   # (alpha, phi, mach, iota)
            self._Fin2_CN    = _psb.get("Fin2_CN",    None)   # (alpha, phi, mach, iota)
            self._Fin2_CBM   = _psb.get("Fin2_CBM",   None)   # (alpha, phi, mach, iota)
            self._Fin2_CHM   = _psb.get("Fin2_CHM",   None)   # (alpha, phi, mach, iota)
            self._Fin4_CN    = _psb.get("Fin4_CN",    None)   # (alpha, phi, mach, iota)
            self._Fin4_CBM   = _psb.get("Fin4_CBM",   None)   # (alpha, phi, mach, iota)
            self._Fin4_CHM   = _psb.get("Fin4_CHM",   None)   # (alpha, phi, mach, iota)


    def _get_CA_Basic(self, alpha: float, phi: float, mach: float, method: str = "linear") -> float:
        return interpn((self.increments.alpha, self.increments.phi, self.increments.mach), #type:ignore
                self._CA_Basic,
                [alpha, phi, mach],
                method=method)[0]
        
    def _get_CA_0_Boost(self, phi: float, mach: float, alt: float, method: str = "linear") -> float:
        return interpn((self.increments.phi, self.increments.mach, self.increments.alt), #type:ignore
                self._CA_0_Boost,
                [phi, mach, alt],
                method=method)[0]


    def _get_CA_0_Coast(self, phi: float, mach: float, alt: float, method: str = "linear") -> float:
        return interpn((self.increments.phi, self.increments.mach, self.increments.alt), #type:ignore
                self._CA_0_Coast,
                [phi, mach, alt],
                method=method)[0]


    def _get_CA_IT(self, alpha: float, phi: float, mach: float, iota: float, method: str = "linear") -> float:
        return interpn((self.increments.alpha, self.increments.phi, self.increments.mach, self.increments.iota), #type:ignore
                self._CA_IT,
                [alpha, phi, mach, iota],
                method=method)[0]


    def _get_CNB_Basic(self, alpha: float, phi: float, mach: float, method: str = "linear") -> float:
        return interpn((self.increments.alpha, self.increments.phi, self.increments.mach), #type:ignore
                self._CNB_Basic,
                [alpha, phi, mach],
                method=method)[0]


    def _get_CNB_IT(self, alpha: float, phi: float, mach: float, iota: float, method: str = "linear") -> float:
        return interpn((self.increments.alpha, self.increments.phi, self.increments.mach, self.increments.iota), #type:ignore
                self._CNB_IT,
                [alpha, phi, mach, iota],
                method=method)[0]


    def _get_CYB_Basic(self, alpha: float, phi: float, mach: float, method: str = "linear") -> float:
        return interpn((self.increments.alpha, self.increments.phi, self.increments.mach), #type:ignore
                self._CYB_Basic,
                [alpha, phi, mach],
                method=method)[0]


    def _get_CYB_IT(self, alpha: float, phi: float, mach: float, iota: float, method: str = "linear") -> float:
        return interpn((self.increments.alpha, self.increments.phi, self.increments.mach, self.increments.iota), #type:ignore
                self._CYB_IT,
                [alpha, phi, mach, iota],
                method=method)[0]


    def _get_CLMB_Basic(self, alpha: float, phi: float, mach: float, method: str = "linear") -> float:
        return interpn((self.increments.alpha, self.increments.phi, self.increments.mach), #type:ignore
                self._CLMB_Basic,
                [alpha, phi, mach],
                method=method)[0]


    def _get_CLMB_IT(self, alpha: float, phi: float, mach: float, iota: float, method: str = "linear") -> float:
        return interpn((self.increments.alpha, self.increments.phi, self.increments.mach, self.increments.iota), #type:ignore
                self._CLMB_IT,
                [alpha, phi, mach, iota],
                method=method)[0]


    def _get_CLNB_Basic(self, alpha: float, phi: float, mach: float, method: str = "linear") -> float:
        return interpn((self.increments.alpha, self.increments.phi, self.increments.mach), #type:ignore
                self._CLNB_Basic,
                [alpha, phi, mach],
                method=method)[0]


    def _get_CLNB_IT(self, alpha: float, phi: float, mach: float, iota: float, method: str = "linear") -> float:
        return interpn((self.increments.alpha, self.increments.phi, self.increments.mach, self.increments.iota), #type:ignore
                self._CLNB_IT,
                [alpha, phi, mach, iota],
                method=method)[0]


    def get_CA_Boost_Total(self, alpha: float, phi: float, mach: float, alt: float, iota: float, method: str = "linear") -> float:
        return self._get_CA_Basic(alpha, phi, mach, method=method)\
                + self._get_CA_0_Boost(phi, mach, alt, method=method)\
                + self._get_CA_IT(alpha, phi, mach, iota, method=method)


    def get_CA_Coast_Total(self, alpha: float, phi: float, mach: float, alt: float, iota: float, method: str = "linear") -> float:
        return self._get_CA_Basic(alpha, phi, mach, method=method)\
                + self._get_CA_0_Coast(phi, mach, alt, method=method)\
                + self._get_CA_IT(alpha, phi, mach, iota, method=method)


    def get_CNB_Total(self, alpha: float, phi: float, mach: float, iota: float, method: str = "linear") -> float:
        return self._get_CNB_Basic(alpha, phi, mach, method=method)\
                + self._get_CNB_IT(alpha, phi, mach, iota, method=method)


    def get_CLMB_Total(self, alpha: float, phi: float, mach: float, iota: float, method: str = "linear") -> float:
        return self._get_CLMB_Basic(alpha, phi, mach, method=method)\
                + self._get_CLMB_IT(alpha, phi, mach, iota, method=method)


    def get_CLNB_Total(self, alpha: float, phi: float, mach: float, iota: float, method: str = "linear") -> float:
        return self._get_CLNB_Basic(alpha, phi, mach, method=method)\
                + self._get_CLNB_IT(alpha, phi, mach, iota, method=method)


    def get_CYB_Total(self, alpha: float, phi: float, mach: float, iota: float, method: str = "linear") -> float:
        return self._get_CYB_Basic(alpha, phi, mach, method=method)\
                + self._get_CYB_IT(alpha, phi, mach, iota, method=method)


    def __repr__(self) -> str:
        members = [i for i in dir(self) if "__" not in i]
        return "\n".join(members)


# if __name__ == "__main__":
#     t = AeroTable("/home/david/work_projects/control/aeromodel/aeromodel.pickle")
#     import time

#     N = 10000
#     st = time.time()
#     for i in range(N):
#         t.get_CA_Boost_Total(1, 0, .3, 1000, 0)
#         t.get_CNB_Total(1, 0, .3, 0)
#         t.get_CYB_Total(1, 0, .3, 0)
#         t.get_CLMB_Total(1, 0, .3, 0)
#         t.get_CLNB_Total(1, 0, .3, 0)
#     print(1/((time.time() - st) / N))
