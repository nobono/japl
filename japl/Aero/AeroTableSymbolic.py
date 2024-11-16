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
                "aerotable.get_CA": self.aerotable.get_CA,
                "aerotable.get_CA_Boost": self.aerotable.get_CA_Boost,
                "aerotable.get_CA_Coast": self.aerotable.get_CA_Coast,
                "aerotable.get_CNB": self.aerotable.get_CNB,
                "aerotable.get_CLMB": self.aerotable.get_CLMB,
                "aerotable.get_CLNB": self.aerotable.get_CLNB,
                "aerotable.get_CYB": self.aerotable.get_CYB,
                "aerotable.get_MRC": self.aerotable.get_MRC,
                "aerotable.get_Sref": self.aerotable.get_Sref,
                "aerotable.get_Lref": self.aerotable.get_Lref,
                "aerotable.get_CA_Boost_alpha": self.aerotable.get_CA_Boost_alpha,
                "aerotable.get_CA_Coast_alpha": self.aerotable.get_CA_Coast_alpha,
                "aerotable.get_CNB_alpha": self.aerotable.get_CNB_alpha,
                "aerotable.inv_aerodynamics": self.aerotable.inv_aerodynamics,
                }
        self.get_CA = Function("aerotable.get_CA")
        self.get_CA_Boost = Function("aerotable.get_CA_Boost")
        self.get_CA_Coast = Function("aerotable.get_CA_Coast")
        self.get_CNB = Function("aerotable.get_CNB")
        self.get_CLMB = Function("aerotable.get_CLMB")
        self.get_CLNB = Function("aerotable.get_CLNB")
        self.get_CYB = Function("aerotable.get_CYB")
        self.get_MRC = Function("aerotable.get_MRC")
        self.get_Sref = Function("aerotable.get_Sref")
        self.get_Lref = Function("aerotable.get_Lref")
        self.get_CA_Boost_alpha = Function("aerotable.get_CA_Boost_alpha")
        self.get_CA_Coast_alpha = Function("aerotable.get_CA_Coast_alpha")
        self.get_CNB_alpha = Function("aerotable.get_CNB_alpha")
        self.inv_aerodynamics = Function("aerotable.inv_aerodynamics")
