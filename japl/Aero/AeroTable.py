import numpy as np
from scipy.interpolate import interpn
from scipy.io import loadmat
import pickle
from astropy import units as u

from japl.Util.Matlab import MatFile



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

    __ft2m = (1.0 * u.imperial.foot).to_value(u.m) #type:ignore
    __inch2m = (1.0 * u.imperial.inch).to_value(u.m) #type:ignore
    __deg2rad = (np.pi / 180.0)
    __lbminch2Nm = (1.0 * u.imperial.lbm * u.imperial.inch**2).to_value(u.kg * u.m**2) #type:ignore
    __inch_sq_2_m_sq = (1.0 * u.imperial.inch**2).to_value(u.m**2) #type:ignore

    def __init__(self, data: str|dict|MatFile) -> None:
        data_dict = {}
        if isinstance(data, str):
            self.__path = data
            if ".pickle" in self.__path:
                with open(self.__path, "rb") as f:
                    data_dict = pickle.load(f)
            elif ".mat" in self.__path:
                data_dict = MatFile(self.__path)
            
        self.increments = Increments()
        self.increments.alpha       = data_dict.get("Alpha", None)
        self.increments.phi         = data_dict.get("Phi", None)
        self.increments.mach        = data_dict.get("Mach", None)
        self.increments.alt         = data_dict.get("Alt", None)
        self.increments.iota        = data_dict.get("Iota", None)
        self.increments.iota_prime  = data_dict.get("Iota_Prime", None)
        try:
            self.increments.nalpha       = len(self.increments.alpha)
            self.increments.nphi         = len(self.increments.phi)
            self.increments.nmach        = len(self.increments.mach)
            self.increments.nalt         = len(self.increments.alt)
            self.increments.niota        = len(self.increments.iota)
            self.increments.niota_prime  = len(self.increments.iota_prime)
        except:
            pass

        self._CA_inv     = data_dict.get("CA_inv",     None).table   # (alpha, phi, mach)
        self._CA_Basic   = data_dict.get("CA_Basic",   None).table   # (alpha, phi, mach)
        self._CA_0_Boost = data_dict.get("CA_0_Boost", None).table   # (phi, mach, alt)
        self._CA_0_Coast = data_dict.get("CA_0_Coast", None).table   # (phi, mach, alt)
        self._CA_IT      = data_dict.get("CA_IT",      None).table   # (alpha, phi, mach, iota)
        self._CYB_Basic  = data_dict.get("CYB_Basic",  None).table   # (alpha, phi, mach)
        self._CYB_IT     = data_dict.get("CYB_IT",     None).table   # (alpha, phi, mach, iota)
        self._CNB_Basic  = data_dict.get("CNB_Basic",  None).table   # (alpha, phi, mach)
        self._CNB_IT     = data_dict.get("CNB_IT",     None).table   # (alpha, phi, mach, iota)
        self._CLLB_Basic = data_dict.get("CLLB_Basic", None).table   # (alpha, phi, mach)
        self._CLLB_IT    = data_dict.get("CLLB_IT",    None).table   # (alpha, phi, mach, iota)
        self._CLMB_Basic = data_dict.get("CLMB_Basic", None).table   # (alpha, phi, mach)
        self._CLMB_IT    = data_dict.get("CLMB_IT",    None).table   # (alpha, phi, mach, iota)
        self._CLNB_Basic = data_dict.get("CLNB_Basic", None).table   # (alpha, phi, mach)
        self._CLNB_IT    = data_dict.get("CLNB_IT",    None).table   # (alpha, phi, mach, iota)
        self._Fin2_CN    = data_dict.get("Fin2_CN",    None).table   # (alpha, phi, mach, iota)
        self._Fin2_CBM   = data_dict.get("Fin2_CBM",   None).table   # (alpha, phi, mach, iota)
        self._Fin2_CHM   = data_dict.get("Fin2_CHM",   None).table   # (alpha, phi, mach, iota)
        self._Fin4_CN    = data_dict.get("Fin4_CN",    None).table   # (alpha, phi, mach, iota)
        self._Fin4_CBM   = data_dict.get("Fin4_CBM",   None).table   # (alpha, phi, mach, iota)
        self._Fin4_CHM   = data_dict.get("Fin4_CHM",   None).table   # (alpha, phi, mach, iota)

        self.Sref = data_dict.get("Sref", None)
        self.Lref = data_dict.get("Lref", None)
        self.MRC = data_dict.get("MRC", None)

        ############################################################
        # TODO make input and ouput of units better...
        # TEMP: currently aero data available is in imperial units
        ############################################################
        self.increments.alpha       = self.increments.alpha      * self.__deg2rad
        self.increments.phi         = self.increments.phi        * self.__deg2rad
        self.increments.alt         = self.increments.alt        * self.__ft2m
        self.increments.iota        = self.increments.iota       * self.__deg2rad
        self.increments.iota_prime  = self.increments.iota_prime * self.__deg2rad
        self.Sref = self.__inch_sq_2_m_sq
        self.Lref = self.__inch2m
        self.MRC = data_dict.get("MRC", None)
        ############################################################



    def _get_CA_Basic(self, alpha: float, phi: float, mach: float, method: str = "linear") -> float:
        return interpn((self.increments.alpha, self.increments.phi, self.increments.mach), #type:ignore
                self._CA_Basic,
                [abs(alpha), phi, mach],
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
                [abs(alpha), phi, mach, iota],
                method=method)[0]


    def _get_CNB_Basic(self, alpha: float, phi: float, mach: float, method: str = "linear") -> float:
        if alpha < 0:
            return -interpn((self.increments.alpha, self.increments.phi, self.increments.mach), #type:ignore
                    self._CNB_Basic,
                    [-alpha, phi, mach],
                    method=method)[0]
        else:
            return interpn((self.increments.alpha, self.increments.phi, self.increments.mach), #type:ignore
                    self._CNB_Basic,
                    [alpha, phi, mach],
                    method=method)[0]


    def _get_CNB_IT(self, alpha: float, phi: float, mach: float, iota: float, method: str = "linear") -> float:
        if alpha < 0:
            return -interpn((self.increments.alpha, self.increments.phi, self.increments.mach, self.increments.iota), #type:ignore
                    self._CNB_IT,
                    [-alpha, phi, mach, -iota],
                    method=method)[0]
        else:
            return interpn((self.increments.alpha, self.increments.phi, self.increments.mach, self.increments.iota), #type:ignore
                    self._CNB_IT,
                    [alpha, phi, mach, iota],
                    method=method)[0]


    def _get_CYB_Basic(self, alpha: float, phi: float, mach: float, method: str = "linear") -> float:
        return interpn((self.increments.alpha, self.increments.phi, self.increments.mach), #type:ignore
                self._CYB_Basic,
                [abs(alpha), phi, mach],
                method=method)[0]


    def _get_CYB_IT(self, alpha: float, phi: float, mach: float, iota: float, method: str = "linear") -> float:
        return interpn((self.increments.alpha, self.increments.phi, self.increments.mach, self.increments.iota), #type:ignore
                self._CYB_IT,
                [abs(alpha), phi, mach, iota],
                method=method)[0]


    def _get_CLMB_Basic(self, alpha: float, phi: float, mach: float, method: str = "linear") -> float:
        if alpha < 0:
            return -interpn((self.increments.alpha, self.increments.phi, self.increments.mach), #type:ignore
                    self._CLMB_Basic,
                    [-alpha, phi, mach],
                    method=method)[0]
        else:
            return interpn((self.increments.alpha, self.increments.phi, self.increments.mach), #type:ignore
                    self._CLMB_Basic,
                    [alpha, phi, mach],
                    method=method)[0]


    def _get_CLMB_IT(self, alpha: float, phi: float, mach: float, iota: float, method: str = "linear") -> float:
        if alpha < 0:
            return -interpn((self.increments.alpha, self.increments.phi, self.increments.mach, self.increments.iota), #type:ignore
                    self._CLMB_IT,
                    [-alpha, phi, mach, -iota],
                    method=method)[0]
        else:
            return interpn((self.increments.alpha, self.increments.phi, self.increments.mach, self.increments.iota), #type:ignore
                    self._CLMB_IT,
                    [alpha, phi, mach, iota],
                    method=method)[0]


    def _get_CLNB_Basic(self, alpha: float, phi: float, mach: float, method: str = "linear") -> float:
        return interpn((self.increments.alpha, self.increments.phi, self.increments.mach), #type:ignore
                self._CLNB_Basic,
                [abs(alpha), phi, mach],
                method=method)[0]


    def _get_CLNB_IT(self, alpha: float, phi: float, mach: float, iota: float, method: str = "linear") -> float:
        return interpn((self.increments.alpha, self.increments.phi, self.increments.mach, self.increments.iota), #type:ignore
                self._CLNB_IT,
                [abs(alpha), phi, mach, iota],
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


    def get_MRC(self) -> float:
        return self.MRC[0]


    def get_Sref(self) -> float:
        return self.Sref


    def get_Lref(self) -> float:
        return self.Lref


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
