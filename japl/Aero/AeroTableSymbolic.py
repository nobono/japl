from typing import Optional
from sympy import Function
from japl import AeroTable
from japl.Util.Matlab import MatFile



class AeroTableSymbolic:

    """This is the Symbolic mirror of the AeroTable module
    which can be used for creating models from symblic expressions."""


    def __init__(self, data: Optional[str|dict|MatFile] = None, from_template: str = "", units: str = "si") -> None:
        self.aerotable = AeroTable(data, from_template=from_template, units=units)
        self.modules = {
                "aerotable_get_CA": self.aerotable.get_CA,
                "aerotable_get_CA_Boost": self.aerotable.get_CA_Boost,
                "aerotable_get_CA_Coast": self.aerotable.get_CA_Coast,
                "aerotable_get_CNB": self.aerotable.get_CNB,
                "aerotable_get_CLMB": self.aerotable.get_CLMB,
                "aerotable_get_CLNB": self.aerotable.get_CLNB,
                "aerotable_get_CYB": self.aerotable.get_CYB,
                "aerotable_get_MRC": self.aerotable.get_MRC,
                "aerotable_get_Sref": self.aerotable.get_Sref,
                "aerotable_get_Lref": self.aerotable.get_Lref,
                "aerotable_get_CA_Boost_alpha": self.aerotable.get_CA_Boost_alpha,
                "aerotable_get_CA_Coast_alpha": self.aerotable.get_CA_Coast_alpha,
                "aerotable_get_CNB_alpha": self.aerotable.get_CNB_alpha,
                "aerotable_inv_aerodynamics": self.aerotable.inv_aerodynamics,
                }
        self.get_CA = Function("aerotable_get_CA")
        self.get_CA_Boost = Function("aerotable_get_CA_Boost")
        self.get_CA_Coast = Function("aerotable_get_CA_Coast")
        self.get_CNB = Function("aerotable_get_CNB")
        self.get_CLMB = Function("aerotable_get_CLMB")
        self.get_CLNB = Function("aerotable_get_CLNB")
        self.get_CYB = Function("aerotable_get_CYB")
        self.get_MRC = Function("aerotable_get_MRC")
        self.get_Sref = Function("aerotable_get_Sref")
        self.get_Lref = Function("aerotable_get_Lref")
        self.get_CA_Boost_alpha = Function("aerotable_get_CA_Boost_alpha")
        self.get_CA_Coast_alpha = Function("aerotable_get_CA_Coast_alpha")
        self.get_CNB_alpha = Function("aerotable_get_CNB_alpha")
        self.inv_aerodynamics = Function("aerotable_inv_aerodynamics")
