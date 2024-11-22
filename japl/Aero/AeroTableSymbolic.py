from typing import Optional
# from sympy import Function
from sympy import Symbol
from japl import AeroTable
from japl.Util.Matlab import MatFile
from japl.Symbolic.KwargFunction import KwargFunction



class get_CA(KwargFunction):
    parent = "aerotable"


class get_CA_Boost(KwargFunction):
    parent = "aerotable"


class get_CA_Coast(KwargFunction):
    parent = "aerotable"


class get_CNB(KwargFunction):
    parent = "aerotable"


class get_CLMB(KwargFunction):
    parent = "aerotable"


class get_CLNB(KwargFunction):
    parent = "aerotable"


class get_CYB(KwargFunction):
    parent = "aerotable"


class get_MRC(KwargFunction):
    parent = "aerotable"


class get_Sref(KwargFunction):
    parent = "aerotable"


class get_Lref(KwargFunction):
    parent = "aerotable"


class get_CA_Boost_alpha(KwargFunction):
    parent = "aerotable"


class get_CA_Coast_alpha(KwargFunction):
    parent = "aerotable"


class get_CNB_alpha(KwargFunction):
    parent = "aerotable"


class inv_aerodynamics(KwargFunction):
    parent = "aerotable"


class AeroTableSymbolic:

    """This is the Symbolic mirror of the AeroTable module
    which can be used for creating models from symblic expressions."""


    def __init__(self, data: Optional[str|dict|MatFile] = None, from_template: str = "", units: str = "si") -> None:
        self.aerotable = AeroTable(data, from_template=from_template, units=units)
        # self.modules = {
        #         "aerotable.CA": self.aerotable.get_CA,
        #         "aerotable.CA_Boost": self.aerotable.get_CA_Boost,
        #         "aerotable.CA_Coast": self.aerotable.get_CA_Coast,
        #         "aerotable.CNB": self.aerotable.get_CNB,
        #         "aerotable.CLMB": self.aerotable.get_CLMB,
        #         "aerotable.CLNB": self.aerotable.get_CLNB,
        #         "aerotable.CYB": self.aerotable.get_CYB,
        #         "aerotable.MRC": self.aerotable.get_MRC,
        #         "aerotable.Sref": self.aerotable.get_Sref,
        #         "aerotable.Lref": self.aerotable.get_Lref,
        #         "aerotable.CA_Boost_alpha": self.aerotable.get_CA_Boost_alpha,
        #         "aerotable.CA_Coast_alpha": self.aerotable.get_CA_Coast_alpha,
        #         "aerotable.CNB_alpha": self.aerotable.get_CNB_alpha,
        #         "aerotable.inv_aerodynamics": self.aerotable.inv_aerodynamics,
        #         }
        # self.CA = Function("aerotable.CA")
        # self.CA_Boost = Function("aerotable.CA_Boost")
        # self.CA_Coast = Function("aerotable.CA_Coast")
        # self.CNB = Function("aerotable.CNB")
        # self.CLMB = Function("aerotable.CLMB")
        # self.CLNB = Function("aerotable.CLNB")
        # self.CYB = Function("aerotable.CYB")
        # self.MRC = Function("aerotable.MRC")
        # self.Sref = Function("aerotable.Sref")
        # self.Lref = Function("aerotable.Lref")
        # self.CA_Boost_alpha = Function("aerotable.CA_Boost_alpha")
        # self.CA_Coast_alpha = Function("aerotable.CA_Coast_alpha")
        # self.CNB_alpha = Function("aerotable.CNB_alpha")
        # self.inv_aerodynamics = Function("aerotable.inv_aerodynamics")
        self.modules = {}
        self.get_CA = get_CA
        self.get_CA_Boost = get_CA_Boost
        self.get_CA_Coast = get_CA_Coast
        self.get_CNB = get_CNB
        self.get_CLMB = get_CLMB
        self.get_CLNB = get_CLNB
        self.get_CYB = get_CYB
        self.get_MRC = get_MRC
        self.get_Sref = get_Sref
        self.get_Lref = get_Lref
        self.get_CA_Boost_alpha = get_CA_Boost_alpha
        self.get_CA_Coast_alpha = get_CA_Coast_alpha
        self.get_CNB_alpha = get_CNB_alpha
        self.inv_aerodynamics = inv_aerodynamics
